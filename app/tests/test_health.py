from fastapi.testclient import TestClient


def test_health_check_endpoint(client: TestClient):
    """Verify API v1 health endpoint returns status 200 OK."""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data
    assert "environment" in data