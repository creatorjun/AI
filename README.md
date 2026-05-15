# AI 서비스 아키텍처 설계서

작성일: 2026-05-15  
버전: v1.0

***

## 1. 프로젝트 개요

메인 에이전트 — 서브 에이전트 — 저지 에이전트 3계층 멀티 모델 협의체 시스템의 핵심 인프라로서,
RAG(Retrieval-Augmented Generation) 및 인터넷 서치 기능을 독립적으로 배포·테스트 가능한
FastAPI 기반 AI 서비스로 구현한다.

***

## 2. 설계 원칙

### 클린 아키텍처

의존성 방향은 단방향이다: `domain → application → infrastructure`. 상위 레이어가 하위 레이어를
알지 못하며, 하위 레이어의 구체 구현이 교체되어도 상위 레이어 코드는 변경되지 않는다.

```
┌─────────────────────────────────────────────────────┐
│  API Layer        (라우터, 스키마, DI)               │
├─────────────────────────────────────────────────────┤
│  Application Layer (유스케이스 — 비즈니스 흐름)      │
├─────────────────────────────────────────────────────┤
│  Domain Layer     (모델, 포트 인터페이스)             │
├─────────────────────────────────────────────────────┤
│  Infrastructure   (어댑터 — DB, API, 외부 서비스)    │
└─────────────────────────────────────────────────────┘
         의존성 방향: 위 → 아래만 허용
```

### 포트-어댑터 패턴

`domain/ports.py` 에 추상 인터페이스(포트)를 정의하고, `infrastructure/` 에 구체 구현체(어댑터)를 배치한다.
어댑터 교체는 `api/deps.py` DI 설정 한 곳만 수정하면 완결된다.

| 포트 인터페이스 | 현재 어댑터 | 교체 가능 예시 |
|----------------|------------|---------------|
| `IEmbedder` | `OpenAIEmbedder` | CohereEmbedder, LocalEmbedder |
| `IVectorStore` | `PgVectorStore` | QdrantStore, WeaviateStore |
| `IChunker` | `SemanticChunker` | FixedChunker, HierarchicalChunker |
| `IReranker` | `VLLMReranker` | CohereReranker, NoOpReranker |
| `IWebSearcher` | `TavilySearcher` | BraveSearcher, SerperSearcher |

### 독립 배포 가능

FastAPI 서버 없이도 유스케이스를 라이브러리로 직접 import하여 사용할 수 있다.
에이전트 시스템의 동일 프로세스 내 직접 호출과 HTTP API 방식 양쪽을 모두 지원한다.

***

## 3. 기술 스택

| 역할 | 기술 | 비고 |
|------|------|------|
| API 서버 | FastAPI | async 완전 지원 |
| ORM | SQLAlchemy 2.0 (async) | |
| 벡터 저장/검색 | pgvector | PostgreSQL 익스텐션 |
| 풀텍스트 검색 | PostgreSQL tsvector | 한국어: pg_bigm |
| 임베딩 모델 | OpenAI text-embedding-3-large | IEmbedder로 추상화, 교체 가능 |
| LLM 추론 (리랭킹) | vLLM (로컬) | OpenAI 호환 API |
| 청킹 | SemanticChunker (기본) | 코사인 유사도 기반 경계 탐지, 교체 가능 |
| 폴더 감시 | watchdog | 파일 이벤트 → 자동 재임베딩 |
| 인터넷 검색 | Tavily API (기본) | IWebSearcher로 추상화, 교체 가능 |
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

## 4. 폴더 구조

```
.
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
├── evaluation/                             # 품질 검증 파이프라인
│   ├── __init__.py
│   ├── dataset.py                          # 골든 데이터셋 로더
│   ├── ragas_evaluator.py                  # Faithfulness / ContextRecall / AnswerRelevancy
│   ├── experiment_matrix.py                # 비교 실험 매트릭스 실행기
│   ├── report.py                           # CSV/JSON 저장 + 콘솔 리포트
│   └── run_phase5.py                       # 단독 실행 엔트리포인트
│
├── app/
│   ├── main.py                             # lifespan: FolderWatcher 통합
│   ├── config.py
│   ├── database.py
│   │
│   ├── domain/                             # 순수 도메인 — 외부 의존성 Zero
│   │   ├── models.py                       # RAGChunk, SearchRequest, WebSearchRequest 등
│   │   └── ports.py                        # IEmbedder, IVectorStore, IWebSearcher 등
│   │
│   ├── application/                        # 유스케이스 — 비즈니스 흐름 조합
│   │   ├── ingest_usecase.py
│   │   ├── search_usecase.py               # Parent-Child 컨텍스트 확장
│   │   ├── manage_usecase.py
│   │   └── web_search_usecase.py           # 인터넷 검색 + auto_ingest  [Phase 6]
│   │
│   ├── infrastructure/                     # 어댑터 — 구체 구현체
│   │   ├── orm_models.py
│   │   ├── pg_vector_store.py              # hybrid_alpha RRF, get_parent_chunks
│   │   ├── openai_embedder.py
│   │   ├── vllm_reranker.py                # VLLMReranker / NoOpReranker
│   │   ├── chunker.py                      # Fixed / Sentence / Hierarchical / Semantic
│   │   ├── folder_watcher.py               # watchdog 기반 FolderWatcher
│   │   ├── tavily_searcher.py              # TavilySearcher (기본)          [Phase 6]
│   │   ├── brave_searcher.py               # BraveSearcher (대안)           [Phase 6]
│   │   └── serper_searcher.py              # SerperSearcher (대안)          [Phase 6]
│   │
│   └── api/
│       ├── deps.py                         # DI 팩토리 — 어댑터 조립 단일 진입점
│       ├── routers/
│       │   ├── ingest.py
│       │   ├── search.py
│       │   ├── manage.py
│       │   └── web_search.py               # POST /search/web               [Phase 6]
│       └── schemas/
│           ├── ingest_schema.py
│           ├── search_schema.py
│           ├── manage_schema.py
│           └── web_search_schema.py                                         [Phase 6]
│
tests/
├── conftest.py
├── unit/
│   ├── test_chunker.py
│   ├── test_metadata_filter.py
│   ├── test_bitemporal.py
│   ├── test_models.py
│   ├── test_hybrid_rrf.py
│   ├── test_folder_watcher.py
│   ├── test_evaluation.py
│   └── test_web_search.py                  # IWebSearcher mock 기반 단위 테스트  [Phase 6]
└── integration/
    ├── test_ingest.py
    ├── test_search.py
    ├── test_manage_cycle.py
    └── test_web_search_integration.py                                       [Phase 6]
```

***

## 5. 도메인 모델

### RAG 도메인

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

### 인터넷 서치 도메인

```python
class WebSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    auto_ingest: bool = False           # True 시 결과를 RAG에 자동 ingest
    ingest_trust_tier: int = Field(default=2, ge=1, le=5)  # 웹 출처 기본 Tier 2
    ingest_tags: list[str] = []

class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    published_at: datetime | None = None
    source: str                         # "tavily" | "brave" | "serper"
```

***

## 6. 포트 인터페이스

### 기존 RAG 포트

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

### 인터넷 서치 포트 (신규)

```python
class IWebSearcher(ABC):
    async def search(self, request: WebSearchRequest) -> list[WebSearchResult]: ...
    async def fetch_content(self, url: str) -> str: ...
```

`fetch_content` 는 스니펫만으로는 RAG ingest 품질이 낮기 때문에 인터페이스에 포함한다.
`auto_ingest=True` 시 유스케이스가 이 메서드로 전문을 가져와 청킹·임베딩 후 저장한다.

***

## 7. API 엔드포인트 명세

### Ingest

```
POST /ingest/folder
Body: { "folder_path": str, "trust_tier": int, "tags": list[str],
        "valid_from": datetime, "valid_to": datetime | null }
Response: { "queued_files": int, "folder_path": str }

POST /ingest/file
Body: { "file_path": str, "trust_tier": int, "tags": list[str],
        "valid_from": datetime, "valid_to": datetime | null }
Response: { "doc_id": str, "source_path": str }

POST /ingest/sync
Body: { "folder_path": str }
Response: { "added": int, "updated": int, "deleted": int }
```

지원 파일 형식: `.txt` `.md` `.rst` `.csv` `.json`

### Search (RAG 내부)

```
POST /search
Body: { "query": str, "as_of": datetime | null, "trust_tier_min": int | null,
        "tags": list[str] | null, "top_k": int, "search_mode": "hybrid"|"vector"|"fulltext",
        "rerank": bool, "hybrid_alpha": float, "use_parent_context": bool }
Response: { "results": [ SearchResult ], "total": int }
```

### Search (인터넷) [Phase 6]

```
POST /search/web
Body: { "query": str, "top_k": int,
        "auto_ingest": bool, "ingest_trust_tier": int, "ingest_tags": list[str] }
Response: { "results": [ WebSearchResult ], "total": int, "auto_ingested": int }
```

`auto_ingested` 필드는 실제로 RAG에 ingest된 URL 건수를 반환한다.

### Manage

```
POST   /documents
PUT    /documents/{doc_id}
DELETE /documents/{doc_id}
GET    /documents/{doc_id}
```

***

## 8. 인터넷 서치 툴 아키텍처

### 의존성 흐름

```
[POST /search/web]
  → WebSearchSchema          (api/schemas)
  → WebSearchUsecase         (application)
      ├─ IWebSearcher.search()          ← TavilySearcher / BraveSearcher / SerperSearcher
      ├─ IWebSearcher.fetch_content()   ← auto_ingest=True 시
      ├─ IEmbedder.embed_batch()        ← auto_ingest=True 시 (기존 포트 재사용)
      └─ IVectorStore.write_batch()     ← auto_ingest=True 시 (기존 포트 재사용)
```

`WebSearchUsecase` 는 `IngestUsecase` 를 직접 의존하지 않는다. 유스케이스 간 직접 의존은
레이어 규칙 위반이므로, `IEmbedder` 와 `IVectorStore` 를 직접 주입받아 ingest를 수행한다.
신규 의존성은 `IWebSearcher` 하나뿐이며, `deps.py` 에 팩토리 함수 하나만 추가하면 DI가 완결된다.

### 어댑터 비교

| 항목 | TavilySearcher | BraveSearcher | SerperSearcher |
|------|---------------|--------------|---------------|
| 공식 Python SDK | ✅ `tavily-python` | ❌ httpx 직접 | ❌ httpx 직접 |
| `fetch_content` 기본 제공 | ✅ `include_raw_content=True` | ❌ 별도 httpx | ❌ 별도 httpx |
| 무료 플랜 월 쿼리 | 1,000 | 2,000 | 2,500 |
| 한국어 검색 품질 | 중 | 중상 | 상 (Google 기반) |
| RAG 연동 최적화 | ✅ 최적화됨 | 일반 | 일반 |
| **기본값** | **✅ 기본** | 대안 | 대안 |

`SEARCHER_PROVIDER` 환경변수 하나로 어댑터를 교체한다. 기존 `VLLMReranker` / `NoOpReranker`
분기 방식과 완전히 동일한 패턴이다.

### auto_ingest 동작 시퀀스

```
WebSearchUsecase.search(auto_ingest=True)
  1. IWebSearcher.search()          → list[WebSearchResult]
  2. IWebSearcher.fetch_content()   → 각 URL 전문 텍스트 (병렬)
  3. content → RAGChunk 생성
     - doc_id     = sha256(url)[:16]
     - source_path = url
     - trust_tier = request.ingest_trust_tier  (기본 2 — 웹 출처)
     - tags       = request.ingest_tags + ["web", provider]
  4. IEmbedder.embed_batch()        → 임베딩 생성
  5. IVectorStore.write_batch()     → RAG DB 저장
  6. return results, auto_ingested_count
```

***

## 9. DB 스키마

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

`PUT /documents/{doc_id}` 수정 시 기존 청크의 `valid_to = now()` 로 닫고 새 청크를 삽입한다.
전체 수정 이력이 보존되어 `as_of` 파라미터로 특정 시점의 상태를 재현할 수 있다.

### 신뢰 티어 정의

| Tier | 의미 | 예시 |
|------|------|------|
| 5 | 공식 검증 문서 | 공식 API 문서, 법령 |
| 4 | 내부 공식 문서 | 팀 공식 설계서 |
| 3 | 일반 참고 문서 | 기술 블로그, 가이드 |
| 2 | 비공식 / 웹 출처 | 웹 검색 결과, 개인 노트 |
| 1 | 미검증 데이터 | 크롤링 데이터, 실험적 내용 |

***

## 10. 환경변수

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
| SEARCHER_PROVIDER | 인터넷 검색 어댑터 선택 | tavily |
| TAVILY_API_KEY | Tavily API 키 | "" |
| BRAVE_API_KEY | Brave Search API 키 | "" |
| SERPER_API_KEY | Google Serper API 키 | "" |
| SEARCHER_FETCH_CONTENT | auto_ingest 시 전문 수집 여부 | true |

### 기동 명령

```bash
# GPU 없는 개발 모드
docker compose up db api --build

# GPU 포함 전체 기동
docker compose --profile gpu up --build
```

***

## 11. 에이전트 연동 설계

### HTTP API 방식 (외부 마이크로서비스)

```
메인 에이전트 → POST /search          (RAG 검색)
             → POST /search/web       (인터넷 검색 + 선택적 RAG ingest)
             → POST /documents
             → PUT  /documents/{doc_id}
```

### 라이브러리 직접 import 방식 (동일 프로세스)

```python
from app.application.search_usecase import SearchUsecase
from app.application.web_search_usecase import WebSearchUsecase

results = await search_usecase.rag_search(request)
web_results = await web_search_usecase.search(web_request)
```

### 에이전트 자동 RAG 수정 루프

```
메인 에이전트
  └→ rag_search() 호출
  └→ 결과 품질 self-critique
  └→ 품질 미달 시
       ├─ web_search(auto_ingest=True)  ← 인터넷에서 최신 정보 수집 후 RAG 편입
       └─ rag_write() / rag_update()   ← 직접 RAG 수정
  └→ 저지 에이전트에 수정 결과 검증 요청
  └→ 저지 승인 시 확정 / 거부 시 재시도
```

***

## 12. 구현 현황

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 1 | 기반 인프라 (Docker, Alembic, FastAPI lifespan) | ✅ 완료 |
| Phase 2 | 도메인 + 인프라 레이어 (포트, 어댑터, Chunker 4종) | ✅ 완료 |
| Phase 3 | Application + API 레이어 (유스케이스 3종, 라우터, 스키마) | ✅ 완료 |
| Phase 4 | 고도화 (SemanticChunker, RRF 튜닝, FolderWatcher, Parent-Child) | ✅ 완료 |
| Phase 5 | 품질 검증 (RAGAS 평가 파이프라인, 실험 매트릭스) | ✅ 완료 |
| Phase 6 | 인터넷 서치 툴 (IWebSearcher, 어댑터 3종, WebSearchUsecase) | 🔲 예정 |
