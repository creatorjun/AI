# tests/conftest.py
import pytest


@pytest.fixture
def sample_content() -> str:
    return (
        "FastAPI는 Python 기반의 고성능 웹 프레임워크입니다. "
        "비동기 처리를 완전히 지원하며 Pydantic을 통한 데이터 검증을 제공합니다. "
        "OpenAPI 문서를 자동으로 생성해줍니다. "
        "SQLAlchemy 2.0과 함께 사용하면 강력한 ORM 기능을 활용할 수 있습니다. "
        "pgvector 익스텐션을 통해 벡터 검색 기능을 PostgreSQL에서 직접 수행합니다. "
        "클린 아키텍처를 적용하면 도메인 로직과 인프라 코드를 명확히 분리할 수 있습니다. "
        "포트-어댑터 패턴은 인터페이스 교체를 용이하게 합니다."
    )