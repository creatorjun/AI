# RAG 서비스 구현 계획서

작성일: 2026-05-12  
버전: v1.3.0 (Phase 1~5 완료)

***

## 1. 프로젝트 개요

메인 에이전트 - 서브 에이전트 - 저지 에이전트 3계층 멀티 모델 협의체 시스템의 핵심 인프라로서, 독립적으로 배포 및 테스트 가능한 RAG(Retrieval-Augmented Generation) 서비스를 구현한다.

### 설계 원칙

- **클린 아키텍처**: domain → application → infrastructure 단방향 의존성 강제
- **포트-어댑터 패턴**: IEmbedder, IVectorStore, IChunker, IReranker 인터페이스 추상화
- **독립 배포 가능**: FastAPI 서버 없이도 라이브러리로 직접 import 가능
- **점진적 확장**: Phase 단위 빌드업, 각 Phase는 독립적으로 검증 가능

***

## 2. 기술 스택

| 역할 | 기술 | 비고 |
|------|------|------|
| API 서버 | FastAPI | async 완전 지원 |
| ORM | SQLAlchemy 2.0 (async) | |
| 벡터 저장/검색 | pgvector | PostgreSQL 익스텐션 |
| 풀텍스트 검색 | PostgreSQL tsvector | 한국어: pg_bigm |
| 임베딩 모델 | OpenAI text-embedding-3-large | IEmbedder로 추상화, 교체 가능 |
| LLM 추론 (리랭킹) | vLLM (로컬) | OpenAI 호환 API /v1/chat/completions |
| 청킹 | SemanticChunker (기본) | 코사인 유사도 기반 경계 탐지, 교체 가능 |
| 폴더 감시 | watchdog | 파일 이벤트 → 자동 재임베딩 |
| 마이그레이션 | Alembic | |
| 설정 관리 | Pydantic Settings | |
| 테스트 | pytest + httpx | |
| 컨테이너 | Docker Compose | PG + vLLM + API 동시 기동 |

### vLLM 모델 선택 기준 (GPU VRAM)

| VRAM | 권장 모델 |
|------|---------|
| 8GB | Qwen/Qwen2.5-7B-Instruct |
| 16GB | Qwen/Qwen2.5-14B-Instruct |
| 24GB | Qwen/Qwen2.5-32B-Instruct (AWQ 양자화) |

***

## 3. 폴더 구조

```
rag_service/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── pyproject.toml
├── Dockerfile
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 0001_init_rag.py
│
├── evaluation/                         # Phase 5: 품질 검증 파이프라인
│   ├── __init__.py
│   ├── dataset.py                      # 골든 데이터셋 로더
│   ├── ragas_evaluator.py              # Faithfulness / ContextRecall / AnswerRelevancy
│   ├── experiment_matrix.py            # 비교 실험 매트릭스 실행기
│   ├── report.py                       # CSV/JSON 저장 + 콘솔 리포트
│   └── run_phase5.py                   # 단독 실행 엔트리포인트
│
└── app/
    ├── main.py                         # lifespan: FolderWatcher 통합
    ├── config.py
    ├── database.py
    │
    ├── domain/
    │   ├── __init__.py
    │   ├── models.py
    │   └── ports.py
    │
    ├── application/
    │   ├── __init__.py
    │   ├── ingest_usecase.py
    │   ├── search_usecase.py           # Parent-Child 컨텍스트 확장
    │   └── manage_usecase.py
    │
    ├── infrastructure/
    │   ├── __init__.py
    │   ├── orm_models.py
    │   ├── pg_vector_store.py          # hybrid_alpha RRF, get_parent_chunks
    │   ├── openai_embedder.py
    │   ├── vllm_reranker.py
    │   ├── chunker.py                  # SemanticChunker 추가
    │   └── folder_watcher.py           # watchdog 기반 FolderWatcher
    │
    └── api/
        ├── __init__.py
        ├── deps.py
        ├── routers/
        │   ├── __init__.py
        │   ├── ingest.py
        │   ├── search.py
        │   └── manage.py
        └── schemas/
            ├── __init__.py
            ├── ingest_schema.py
            ├── search_schema.py
            └── manage_schema.py

tests/
├── __init__.py
├── conftest.py
├── unit/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_metadata_filter.py
│   ├── test_bitemporal.py
│   ├── test_models.py
│   ├── test_hybrid_rrf.py
│   ├── test_folder_watcher.py
│   └── test_evaluation.py             # Phase 5 평가 파이프라인 테스트
└── integration/
    ├── __init__.py
    ├── test_ingest.py
    ├── test_search.py
    └── test_manage_cycle.py
```

***

## 4. DB 스키마

### 핵심 테이블: rag_documents

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_bigm;

CREATE TABLE rag_documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    parent_doc_id   TEXT,
    content         TEXT NOT NULL,
    embedding       vector(3072),
    content_tsv     tsvector,
    trust_tier      SMALLINT NOT NULL CHECK (trust_tier BETWEEN 1 AND 5),
    tags            TEXT[]   NOT NULL DEFAULT '{}',
    source_path     TEXT     NOT NULL,
    extra_meta      JSONB    NOT NULL DEFAULT '{}',
    valid_from      TIMESTAMPTZ NOT NULL,
    valid_to        TIMESTAMPTZ,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at      TIMESTAMPTZ,

    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX ON rag_documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON rag_documents USING GIN (content_tsv);
CREATE INDEX ON rag_documents USING GIN (tags);
CREATE INDEX ON rag_documents (trust_tier, valid_from, valid_to);
```

### 바이템포럴 설계 원칙

| 필드 | 시간 축 | 설명 |
|------|---------|------|
| valid_from / valid_to | 사실 시간 (Fact Time) | 해당 내용이 현실에서 유효한 기간 |
| recorded_at | 기록 시간 (Record Time) | 시스템에 기록된 시점 |
| deleted_at | — | Soft delete, 물리적 삭제 없음 |

`PUT /documents/{doc_id}` 수정 시 기존 청크의 `valid_to = now()`로 닫고 새 청크를 삽입한다. 전체 수정 이력이 보존되어 `as_of` 파라미터로 특정 시점의 상태를 재현할 수 있다.

### 신뢰 티어 정의

| Tier | 의미 | 예시 |
|------|------|------|
| 5 | 공식 검증 문서 | 공식 API 문서, 법령 |
| 4 | 내부 공식 문서 | 팀 공식 설계서 |
| 3 | 일반 참고 문서 | 기술 블로그, 가이드 |
| 2 | 비공식 메모 | 개인 노트, 초안 |
| 1 | 미검증 데이터 | 크롤링 데이터, 실험적 내용 |

***

## 5. API 엔드포인트 명세

### Ingest

```
POST /ingest/folder
Body: {
    "folder_path": str,
    "trust_tier": int (1~5, default: 3),
    "tags": list[str],
    "valid_from": datetime,
    "valid_to": datetime | null
}
Response: { "queued_files": int, "folder_path": str }

POST /ingest/file
Body: {
    "file_path": str,
    "trust_tier": int,
    "tags": list[str],
    "valid_from": datetime,
    "valid_to": datetime | null
}
Response: { "doc_id": str, "source_path": str }

POST /ingest/sync
Body: { "folder_path": str }
Response: { "added": int, "updated": int, "deleted": int }
```

지원 파일 형식: `.txt` `.md` `.rst` `.csv` `.json`

### Search

```
POST /search
Body: {
    "query": str,
    "as_of": datetime | null,
    "trust_tier_min": int | null,
    "tags": list[str] | null,
    "top_k": int (default: 10),
    "search_mode": "hybrid" | "vector" | "fulltext",
    "rerank": bool (default: true),
    "hybrid_alpha": float (0.0~1.0, default: 0.5),
    "use_parent_context": bool (default: false)
}
Response: {
    "results": [
        {
            "doc_id": str,
            "chunk_index": int,
            "content": str,
            "score": float,
            "trust_tier": int,
            "tags": list[str],
            "valid_from": datetime,
            "valid_to": datetime | null,
            "source_path": str,
            "recorded_at": datetime,
            "parent_content": str | null
        }
    ],
    "total": int
}
```

### 검색 모드 비교

| 모드 | 방식 | 특징 |
|------|------|------|
| vector | pgvector 코사인 유사도 | 의미 기반 검색, 키워드 정확도 낮음 |
| fulltext | PostgreSQL tsvector | 키워드 정확도 높음, 의미 유사도 약함 |
| hybrid | RRF(Reciprocal Rank Fusion) | vector + fulltext 상호 보완, 기본 권장 |

#### hybrid_alpha 튜닝 가이드

| alpha 값 | 특성 |
|---------|------|
| 1.0 | vector 검색 100% (의미 유사도 극대화) |
| 0.5 | vector / fulltext 균등 (기본값, 범용 권장) |
| 0.0 | fulltext 검색 100% (키워드 정확도 극대화) |

### Manage

```
POST   /documents
PUT    /documents/{doc_id}
DELETE /documents/{doc_id}
GET    /documents/{doc_id}
```

***

## 6. 도메인 모델

```python
class RAGDocumentMeta(BaseModel):
    doc_id: str
    chunk_index: int
    parent_doc_id: str | None = None
    source_path: str
    trust_tier: int = Field(..., ge=1, le=5)
    tags: list[str] = []
    extra_meta: dict = {}
    valid_from: datetime
    valid_to: datetime | None = None
    recorded_at: datetime | None = None

class RAGChunk(BaseModel):
    meta: RAGDocumentMeta
    content: str
    embedding: list[float] | None = None

class SearchRequest(BaseModel):
    query: str
    as_of: datetime | None = None
    trust_tier_min: int | None = None
    tags: list[str] | None = None
    top_k: int = 10
    search_mode: Literal["hybrid", "vector", "fulltext"] = "hybrid"
    rerank: bool = True
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    use_parent_context: bool = False

class SearchResult(BaseModel):
    doc_id: str
    chunk_index: int
    content: str
    score: float
    trust_tier: int
    tags: list[str]
    valid_from: datetime
    valid_to: datetime | None
    source_path: str
    recorded_at: datetime
    parent_content: str | None = None
```

***

## 7. 포트 인터페이스

```python
class IEmbedder(ABC):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

class IChunker(ABC):
    def chunk(self, content: str, source_path: str) -> list[str]: ...

class IReranker(ABC):
    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]: ...

class IVectorStore(ABC):
    async def write(self, chunk: RAGChunk) -> str: ...
    async def write_batch(self, chunks: list[RAGChunk]) -> list[str]: ...
    async def update(self, request: UpdateRequest, embedding: list[float]) -> str: ...
    async def delete(self, doc_id: str) -> int: ...
    async def search(
        self, request: SearchRequest, query_embedding: list[float]
    ) -> list[SearchResult]: ...
    async def get_by_doc_id(self, doc_id: str) -> list[RAGChunk]: ...
    async def get_source_paths(self) -> list[str]: ...
    async def get_parent_chunks(self, parent_doc_ids: list[str]) -> dict[str, str]: ...
```

***

## 8. 인프라 설정

### 환경변수 목록

| 변수 | 설명 | 기본값 |
|------|------|--------|
| DATABASE_URL | PostgreSQL 연결 문자열 | 필수 |
| OPENAI_API_KEY | OpenAI 임베딩 API 키 | 필수 |
| EMBEDDING_MODEL | 임베딩 모델명 | text-embedding-3-large |
| EMBEDDING_DIMENSIONS | 벡터 차원 수 | 3072 |
| VLLM_BASE_URL | vLLM API 엔드포인트 | http://vllm:8000/v1 |
| VLLM_MODEL | vLLM 로드 모델명 | Qwen/Qwen2.5-7B-Instruct |
| VLLM_API_KEY | vLLM API 키 | EMPTY |
| HF_TOKEN | HuggingFace 토큰 | 비공개 모델 시 필수 |
| APP_ENV | 환경 구분 | development |
| RERANKER_ENABLED | vLLM 리랭킹 활성화 | true |
| HYBRID_ALPHA | RRF vector/fulltext 가중치 (0.0~1.0) | 0.5 |
| WATCH_FOLDER | 자동 임베딩 감시 폴더 경로 | "" (비활성) |
| SEMANTIC_CHUNKER_THRESHOLD | SemanticChunker 경계 임계값 | 0.85 |

### GPU 없는 개발 모드

```cmd
docker compose up db api --build
```

GPU 포함 전체 기동:

```cmd
docker compose --profile gpu up --build
```

***

## 9. 구현 단계 계획

### ✅ Phase 1 — 기반 인프라 `[완료]`

**기간**: 1~2일 | **완료일**: 2026-05-12

| 작업 | 파일 | 상태 |
|------|------|------|
| Docker Compose 구성 | docker-compose.yml | ✅ |
| 환경변수 설정 | .env.example / app/config.py | ✅ |
| AsyncEngine 설정 | app/database.py | ✅ |
| ORM 테이블 정의 | app/infrastructure/orm_models.py | ✅ |
| Alembic 마이그레이션 | alembic/versions/0001_init_rag.py | ✅ |
| FastAPI 진입점 | app/main.py | ✅ |

***

### ✅ Phase 2 — 도메인 + 인프라 레이어 `[완료]`

**기간**: 2~3일 | **완료일**: 2026-05-12

| 작업 | 파일 | 상태 |
|------|------|------|
| 도메인 모델 정의 | app/domain/models.py | ✅ |
| 포트 인터페이스 정의 | app/domain/ports.py | ✅ |
| OpenAI 임베더 구현 | app/infrastructure/openai_embedder.py | ✅ |
| vLLM 리랭커 구현 | app/infrastructure/vllm_reranker.py | ✅ |
| Chunker 구현 | app/infrastructure/chunker.py | ✅ |
| PgVectorStore 구현 | app/infrastructure/pg_vector_store.py | ✅ |
| 단위 테스트 작성 | tests/unit/ | ✅ |

**테스트 결과**: `pytest tests/unit/ -v` → **30 passed**

#### Chunker 구현 목록

| 클래스 | 설명 |
|--------|------|
| FixedChunker | 고정 토큰 크기, overlap 지원 |
| SentenceChunker | 문장 단위 분리, overlap_sentences 지원 |
| HierarchicalChunker | Parent-Child 구조, 검색 정밀도 + 컨텍스트 폭 동시 확보 |

***

### ✅ Phase 3 — Application + API 레이어 `[완료]`

**기간**: 2일 | **완료일**: 2026-05-12

| 작업 | 파일 | 상태 |
|------|------|------|
| Ingest 유스케이스 | app/application/ingest_usecase.py | ✅ |
| Manage 유스케이스 | app/application/manage_usecase.py | ✅ |
| Search 유스케이스 | app/application/search_usecase.py | ✅ |
| DI 설정 | app/api/deps.py | ✅ |
| 라우터 3종 | app/api/routers/ | ✅ |
| 스키마 3종 | app/api/schemas/ | ✅ |
| 통합 테스트 | tests/integration/ | ✅ |

**테스트 결과**: `pytest tests/ -v` → **38 passed**

***

### ✅ Phase 4 — 고도화 `[완료]`

**기간**: 1일 | **완료일**: 2026-05-15

| 작업 | 파일 | 상태 |
|------|------|------|
| SemanticChunker 구현 | app/infrastructure/chunker.py | ✅ |
| hybrid_alpha RRF 튜닝 | app/infrastructure/pg_vector_store.py | ✅ |
| FolderWatcher 구현 | app/infrastructure/folder_watcher.py | ✅ |
| lifespan FolderWatcher 통합 | app/main.py | ✅ |
| Parent-Child 검색 | app/application/search_usecase.py | ✅ |
| IVectorStore.get_parent_chunks 추가 | app/domain/ports.py | ✅ |
| SearchRequest/Result 필드 확장 | app/domain/models.py | ✅ |
| SearchAPIRequest/Item 필드 확장 | app/api/schemas/search_schema.py | ✅ |
| 환경변수 3종 추가 | app/config.py | ✅ |
| Phase 4 테스트 추가 | tests/ | ✅ |

**테스트 결과**: `pytest tests/ -v` → **76 passed, 0 warnings**

#### Chunker 구현 목록 (업데이트)

| 클래스 | 설명 |
|--------|------|
| FixedChunker | 고정 토큰 크기, overlap 지원 |
| SentenceChunker | 문장 단위 분리, overlap_sentences 지원 |
| HierarchicalChunker | Parent-Child 구조, 검색 정밀도 + 컨텍스트 폭 동시 확보 |
| SemanticChunker | 임베딩 코사인 유사도 기반 경계 탐지, Phase 4 기본값 |

***

### ✅ Phase 5 — 품질 검증 `[완료]`

**기간**: 1일 | **완료일**: 2026-05-15

| 작업 | 파일 | 상태 |
|------|------|------|
| 골든 데이터셋 로더 | evaluation/dataset.py | ✅ |
| RAGAS 평가기 구현 | evaluation/ragas_evaluator.py | ✅ |
| 실험 매트릭스 실행기 | evaluation/experiment_matrix.py | ✅ |
| 결과 리포트 (CSV/JSON) | evaluation/report.py | ✅ |
| 단독 실행 엔트리포인트 | evaluation/run_phase5.py | ✅ |
| Phase 5 단위 테스트 | tests/unit/test_evaluation.py | ✅ |

**테스트 결과**: `pytest tests/unit/test_evaluation.py -v` → **18 passed**  
**누적 테스트**: `pytest tests/ -v` → **94 passed, 0 warnings** (예상)

#### 측정 메트릭

| 메트릭 | 설명 | 구현 방식 |
|--------|------|-----------|
| Faithfulness | 답변이 검색 컨텍스트에 충실한 비율 | token overlap (answer ↔ best context) |
| Context Recall | 정답에 필요한 청크가 검색된 비율 | ground_truth_contexts hit rate (threshold 0.15) |
| Answer Relevancy | 검색 결과가 쿼리와 관련된 비율 | cosine similarity (question ↔ answer embedding) |

#### 비교 실험 매트릭스

| 변수 | 후보 |
|------|------|
| 검색 모드 | vector / fulltext / hybrid |
| hybrid_alpha | 0.3 / 0.5 / 0.7 |
| 리랭킹 | on / off |
| Parent-Child | on / off |

#### 실행 방법

```bash
# 기본 골든 데이터셋으로 실행
python evaluation/run_phase5.py

# 커스텀 데이터셋 JSON으로 실행
python evaluation/run_phase5.py path/to/dataset.json

# 단위 테스트만 실행 (DB 불필요)
pytest tests/unit/test_evaluation.py -v
```

***

## 10. 에이전트 연동 설계

### HTTP API 방식 (외부 마이크로서비스)

```
메인 에이전트 → POST /search
             → POST /documents
             → PUT  /documents/{doc_id}
```

### 라이브러리 직접 import 방식 (동일 프로세스)

```python
from app.application.search_usecase import SearchUsecase
from app.application.manage_usecase import ManageUsecase

results = await search_usecase.rag_search(request)
await manage_usecase.rag_write(request)
```

### 에이전트 자동 RAG 수정 루프

```
메인 에이전트
  └→ rag_search() 호출
  └→ 결과 품질 self-critique
  └→ 품질 미달 시 rag_write() / rag_update() 직접 호출
  └→ 저지 에이전트에 수정 결과 검증 요청
  └→ 저지 승인 시 확정 / 거부 시 재시도
```
