# CHANGELOG

모든 주요 변경 사항을 기록합니다.  
형식: [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)  
버전 관리: [Semantic Versioning](https://semver.org/lang/ko/)

***

## [v1.2.0] - 2026-05-15

### Added
- `SemanticChunker`: 인접 문장 임베딩 코사인 유사도 기반 경계 탐지 청커 추가 (`app/infrastructure/chunker.py`)
- `FolderWatcher`: watchdog 기반 폴더 감시 → 파일 생성/수정 시 자동 ingest, 삭제 시 soft-delete (`app/infrastructure/folder_watcher.py`)
- `hybrid_alpha` 파라미터: 요청마다 RRF vector/fulltext 가중치 동적 조정 (`SearchRequest`, `pg_vector_store.py`)
- `use_parent_context` 파라미터: 검색된 자식 청크의 부모 전체 텍스트를 `parent_content`에 주입 (`search_usecase.py`)
- `IVectorStore.get_parent_chunks()`: 부모 doc_id 목록으로 전체 텍스트 일괄 조회 인터페이스 추가 (`ports.py`)
- `SearchResult.parent_content` 필드 추가 (`domain/models.py`)
- `app/main.py` lifespan에 `FolderWatcher` 통합 (`WATCH_FOLDER` 설정 시 서버 기동과 함께 자동 시작)
- 환경변수 3종 추가: `HYBRID_ALPHA`, `WATCH_FOLDER`, `SEMANTIC_CHUNKER_THRESHOLD`
- Phase 4 테스트 추가 (`tests/unit/test_hybrid_rrf.py`, `tests/unit/test_folder_watcher.py`)
- 기존 테스트 확장: `TestSemanticChunker` (7 cases), Phase 4 모델 필드, `use_parent_context` 통합 시나리오 (5 cases)

### Changed
- `deps.py`: 기본 청커를 `SemanticChunker`로 교체
- `SearchAPIRequest` / `SearchResultItem` 스키마에 `hybrid_alpha`, `use_parent_context`, `parent_content` 필드 추가
- README.md 버전 v1.1.0 → v1.2.0, Phase 4 완료 반영

### Fixed
- `tests/unit/test_folder_watcher.py`: `AsyncMock` → `MagicMock` 교체로 `RuntimeWarning: coroutine never awaited` 제거
- `tests/unit/test_folder_watcher.py`: `_make_doc_id` 패치 및 `vector_store` 독립 분리로 마지막 warning 제거
- `tests/unit/test_folder_watcher.py`: 미사용 `_SUPPORTED_EXTENSIONS` import 제거

**테스트 결과**: `pytest tests/ -v` → **76 passed, 0 warnings**

***

## [v1.1.0] - 2026-05-12

### Added
- Phase 3: Application + API 레이어 전체 구현
  - `IngestUsecase`, `ManageUsecase`, `SearchUsecase`
  - FastAPI 라우터 3종 (`ingest`, `search`, `manage`)
  - Pydantic 스키마 3종
  - DI 설정 (`deps.py`)
- 통합 테스트 추가 (`tests/integration/`)
- `NoOpReranker`: `RERANKER_ENABLED=false` 시 vLLM 없이 동작 보장

### Changed
- `SearchRequest.search_mode`: `str` → `Literal["hybrid", "vector", "fulltext"]` 강화
- `UpdateRequest`(수정, doc_id 필수) / `WriteRequest`(생성, doc_id 옵션) 모델 분리

**테스트 결과**: `pytest tests/ -v` → **38 passed**

***

## [v1.0.0] - 2026-05-12

### Added
- Phase 1: 기반 인프라 구성
  - Docker Compose (db + api, vLLM은 `profiles: ["gpu"]`로 분리)
  - Alembic 마이그레이션 (`0001_init_rag.py`)
  - AsyncEngine + SessionLocal (`database.py`)
  - ORM 테이블 정의 (`orm_models.py`)
  - Pydantic Settings 기반 환경변수 관리 (`config.py`)
  - FastAPI lifespan 진입점 (`main.py`)
- Phase 2: 도메인 + 인프라 레이어 구현
  - 도메인 모델: `RAGDocumentMeta`, `RAGChunk`, `SearchRequest`, `SearchResult`
  - 포트 인터페이스: `IEmbedder`, `IChunker`, `IReranker`, `IVectorStore`
  - `OpenAIEmbedder`, `VLLMReranker`, `PgVectorStore`
  - Chunker 3종: `FixedChunker`, `SentenceChunker`, `HierarchicalChunker`
  - 바이템포럴 설계: `valid_from` / `valid_to` / `recorded_at`
  - 신뢰 티어 (Tier 1~5) 설계
  - 단위 테스트 (`tests/unit/`)

**테스트 결과**: `pytest tests/unit/ -v` → **30 passed**
