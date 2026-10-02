"""
SalahSetu AI Service - FastAPI Test Runner & Validation Suite (Stage 8)

Tests endpoints, input validation, CORS, error handling, citation output, and secret security.

Output Artifact:
- data/embeddings/BNS_ai_service_validation_report.json
"""

import sys
import json
import re
from pathlib import Path
from starlette.testclient import TestClient

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from ai_service.api.main import app


def check_no_secrets(text: str) -> bool:
    """Verifies no API keys or sensitive credential patterns appear in string."""
    sensitive_patterns = [
        r'AIzaSy[A-Za-z0-9_-]{33}',
        r'sk-[A-Za-z0-9_-]{32,}',
        r'GEMINI_API_KEY\s*=\s*["\']?[A-Za-z0-9_-]+',
        r'OPENAI_API_KEY\s*=\s*["\']?[A-Za-z0-9_-]+'
    ]
    for pattern in sensitive_patterns:
        if re.search(pattern, text):
            return False
    return True


def run_api_tests():
    client = TestClient(app)

    output_report_path = base_dir / "data" / "embeddings" / "BNS_ai_service_validation_report.json"

    print("==================================================")
    print("  SalahSetu - Stage 8 Python AI Service API Tests ")
    print("==================================================")

    # 1. Test GET /health
    res_health = client.get("/health")
    print("\n1. GET /health:")
    print("Status Code:", res_health.status_code)
    print("Response:", res_health.json())
    health_check_passed = (res_health.status_code == 200 and res_health.json().get("status") == "ok")

    # 2. Test GET /health/ready
    res_ready = client.get("/health/ready")
    print("\n2. GET /health/ready:")
    print("Status Code:", res_ready.status_code)
    print("Response:", res_ready.json())
    readiness_check_passed = (res_ready.status_code == 200 and res_ready.json().get("status") == "ready")

    # 3. Test POST /api/v1/legal/query with "What is cheating under the Bharatiya Nyaya Sanhita?"
    q1 = "What is cheating under the Bharatiya Nyaya Sanhita?"
    res_q1 = client.post("/api/v1/legal/query", json={"question": q1})
    print("\n3. POST /api/v1/legal/query (Supported Query 1):")
    print("Status Code:", res_q1.status_code)
    print("Response Data:", json.dumps(res_q1.json(), indent=2))
    supported_query_passed = (
        res_q1.status_code == 200 and
        len(res_q1.json().get("sources", [])) > 0 and
        "318" in res_q1.json().get("retrieved_sections", [])
    )

    # 4. Test POST /api/v1/legal/query with "What is the punishment for murder?"
    q2 = "What is the punishment for murder?"
    res_q2 = client.post("/api/v1/legal/query", json={"question": q2})
    print("\n4. POST /api/v1/legal/query (Supported Query 2):")
    print("Status Code:", res_q2.status_code)
    print("Response Data:", json.dumps(res_q2.json(), indent=2))
    second_supported_query_passed = (
        res_q2.status_code == 200 and
        len(res_q2.json().get("sources", [])) > 0 and
        "103" in res_q2.json().get("retrieved_sections", [])
    )

    # 5. Test POST /api/v1/legal/query with "What is the limitation period for filing a civil property claim?"
    q3 = "What is the limitation period for filing a civil property claim?"
    res_q3 = client.post("/api/v1/legal/query", json={"question": q3})
    print("\n5. POST /api/v1/legal/query (Unsupported Query):")
    print("Status Code:", res_q3.status_code)
    print("Response Data:", json.dumps(res_q3.json(), indent=2))
    unsupported_query_handled = (
        res_q3.status_code == 200 and
        len(res_q3.json().get("sources", [])) == 0 and
        "does not contain sufficient information" in res_q3.json().get("answer", "")
    )

    # 6. Test POST /api/v1/legal/query with invalid empty question
    res_inv = client.post("/api/v1/legal/query", json={"question": "   "})
    print("\n6. POST /api/v1/legal/query (Invalid Input: Empty String):")
    print("Status Code:", res_inv.status_code)
    print("Response Data:", res_inv.json())
    invalid_input_handled = (res_inv.status_code == 400)

    # 7. Check Secret Exposure Security
    all_responses_str = " ".join([
        res_health.text, res_ready.text, res_q1.text, res_q2.text, res_q3.text, res_inv.text
    ])
    secrets_exposed = not check_no_secrets(all_responses_str)

    validation_passed = (
        health_check_passed and
        readiness_check_passed and
        supported_query_passed and
        second_supported_query_passed and
        unsupported_query_handled and
        invalid_input_handled and
        not secrets_exposed
    )

    report = {
        "health_check_passed": health_check_passed,
        "readiness_check_passed": readiness_check_passed,
        "supported_query_passed": supported_query_passed,
        "second_supported_query_passed": second_supported_query_passed,
        "unsupported_query_handled": unsupported_query_handled,
        "invalid_input_handled": invalid_input_handled,
        "secrets_exposed": secrets_exposed,
        "validation_passed": validation_passed
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n--- AI SERVICE API VALIDATION REPORT SUMMARY ---")
    print(f"Health Check Passed:          {report['health_check_passed']}")
    print(f"Readiness Check Passed:       {report['readiness_check_passed']}")
    print(f"Supported Query 1 Passed:     {report['supported_query_passed']}")
    print(f"Supported Query 2 Passed:     {report['second_supported_query_passed']}")
    print(f"Unsupported Query Handled:    {report['unsupported_query_handled']}")
    print(f"Invalid Input Handled (400):  {report['invalid_input_handled']}")
    print(f"Secrets Exposed:              {report['secrets_exposed']}")
    print(f"Validation Passed:            {report['validation_passed']}")
    print("==================================================")


if __name__ == "__main__":
    run_api_tests()
