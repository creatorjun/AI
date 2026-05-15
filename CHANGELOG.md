# CHANGELOG

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — 2026-05-15

### Performance

#### 🔴 Critical

- **`app/api/deps.py`** — `OpenAIEmbedder`, `VLLMReranker`, `SentenceChunker`를 매 요청마다 재생성하던 문제 수정
  - 기존: `get_embedder()`, `get_reranker()`가 `Depends` 체인 내에서 매 요청마다 호출되어 `AsyncOpenAI` 클라이언트(내부 `httpx.AsyncClient` + TCP 커넥션 풀 포함)가 매번 새로 생성됨
  - 변경: 모듈 레벨 싱글톤 `_embedder`, `_reranker`, `_chunker`로 교체하여 커넥션 풀 재사용 및 인스턴스 생성 오버헤드 제거

- **`app/infrastructure/pg_vector_store.py` — `_vector_search` embedding 문자열 직렬화 제거**
  - 기존: `"[" + ",".join(str(v) for v in embedding) + "]"` 로 3072차원 벡터를 매 쿼리마다 약 30,000자 문자열로 직렬화한 뒤 raw `text()` SQL에 삽입
  - 변경: `pgvector.sqlalchemy` 컬럼의 `.cosine_distance(embedding)` ORM 메서드를 사용, SQLAlchemy 바인드 파라미터로 직접 전달하여 문자열 직렬화 오버헤드 및 SQL Injection 위험 제거

#### 🟡 Major

- **`app/infrastructure/pg_vector_store.py` — `write_batch` N-루프 flush 제거**
  - 기존: `write_batch()`가 내부적으로 `write()`를 N번 순차 호출하며 각 호출마다 `session.flush()` 대기, 청크 수만큼 DB 왕복 발생
  - 변경: `_build_orm()` 헬퍼 메서드 도입 후 `session.add_all(orm_list)` + 단일 `flush()`로 교체, DB 왕복을 1회로 축소

- **`app/database.py` — 커넥션 풀 파라미터 명시**
  - 기존: SQLAlchemy 기본값(`pool_size=5`)으로 동작하여 동시 요청 급증 시 커넥션 고갈 위험 존재
  - 변경: `pool_size=10`, `max_overflow=20`, `pool_timeout=30`, `pool_recycle=1800` 명시적 설정

- **`app/infrastructure/vllm_reranker.py` — `asyncio` 모듈 레벨 import 이동**
  - 기존: `rerank()` 메서드 내부에서 `import asyncio`를 매 호출마다 실행
  - 변경: 파일 최상단 모듈 레벨 import로 이동

- **`app/infrastructure/openai_embedder.py` — `AsyncOpenAI` 클라이언트 싱글톤화**
  - 기존: `OpenAIEmbedder.__init__()` 내에서 매번 `AsyncOpenAI()` 인스턴스 생성
  - 변경: 모듈 레벨 `_client` 싱글톤으로 추출, 인스턴스는 참조만 보유

- **`app/infrastructure/vllm_reranker.py` — `AsyncOpenAI` 클라이언트 싱글톤화**
  - 기존: `VLLMReranker.__init__()` 내에서 매번 `AsyncOpenAI()` 인스턴스 생성
  - 변경: 모듈 레벨 `_client` 싱글톤으로 추출

#### 🟢 Minor

- **`app/main.py` — `/health` 엔드포인트 세션 획득 방식 표준화**
  - 기존: `AsyncSessionLocal()`을 직접 생성하여 `get_db` 의존성 우회, commit/rollback 없이 커넥션 반납
  - 변경: `Depends(get_db)`를 통해 표준 세션 라이프사이클 적용

- **`app/infrastructure/pg_vector_store.py` — `_build_orm()` 헬퍼 메서드 도입**
  - `write()`와 `write_batch()` 양쪽에서 중복되던 ORM 객체 생성 로직을 단일 메서드로 추출

---

## [0.1.0] — 2026-05-12

### Added

- **RAG Service 초기 구현** (`first commit` / `alembic`)
  - Clean Architecture 기반 레이어 구조 수립: `api` → `application` → `domain` → `infrastructure`
  - **Domain 레이어**: `RAGChunk`, `RAGDocumentMeta`, `SearchRequest`, `SearchResult`, `UpdateRequest` 도메인 모델 정의
  - **Domain Ports**: `IEmbedder`, `IChunker`, `IVectorStore`, `IReranker` 추상 인터페이스 정의
  - **Infrastructure 레이어**:
    - `OpenAIEmbedder`: `text-embedding-3-large` (3072차원) 기반 단건/배치 임베딩
    - `PgVectorStore`: pgvector 기반 vector / fulltext / hybrid(RRF) 검색, soft-delete, temporal validity 지원
    - `SentenceChunker` / `FixedChunker` / `HierarchicalChunker`: 문장 단위 / 고정 토큰 / 계층적 청킹
    - `VLLMReranker`: vLLM Chat Completions API 기반 LLM 리랭커 (JSON score 출력)
    - `NoOpReranker`: 리랭커 비활성화 시 pass-through 구현
  - **ORM**: `RAGDocumentORM` — pgvector `Vector(3072)`, `TSVECTOR`, `JSONB`, `ARRAY(Text)`, temporal 컬럼 포함
  - **API 라우터**:
    - `POST /ingest` — 단건/배치 문서 수집 및 청킹·임베딩
    - `POST /search` — vector / fulltext / hybrid 검색 + 리랭킹
    - `GET|PUT|DELETE /manage/{doc_id}` — 문서 조회·수정·삭제 관리
  - **Alembic**: DB 마이그레이션 초기 설정
  - **Docker**: `Dockerfile` + `docker-compose.yml` (PostgreSQL + pgvector, vLLM 서비스 포함)
  - **Config**: `pydantic-settings` 기반 환경변수 관리 (`Settings`)
