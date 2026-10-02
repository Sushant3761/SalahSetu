/**
 * SalahSetu Node.js Backend Stage 9 Test Suite
 */

const http = require('http');

const BACKEND_PORT = 5000;
const BASE_URL = `http://127.0.0.1:${BACKEND_PORT}`;

function makeRequest(method, path, body = null) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, BASE_URL);
    const options = {
      method,
      hostname: url.hostname,
      port: url.port,
      path: url.pathname + url.search,
      headers: {
        'Content-Type': 'application/json'
      }
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve({ status: res.statusCode, data: parsed });
        } catch (e) {
          resolve({ status: res.statusCode, raw: data });
        }
      });
    });

    req.on('error', (err) => reject(err));

    if (body) {
      req.write(JSON.stringify(body));
    }
    req.end();
  });
}

async function runTests() {
  console.log("==================================================");
  console.log("  SalahSetu - Stage 9 Node.js Backend Test Suite  ");
  console.log("==================================================");

  const results = {};

  // 1. GET /api/health
  try {
    const res1 = await makeRequest('GET', '/api/health');
    console.log('\n1. GET /api/health:');
    console.log('Status Code:', res1.status);
    console.log('Response:', res1.data);
    results.healthCheck = (res1.status === 200 && res1.data.status === 'ok');
  } catch (e) {
    console.error('Health check failed:', e.message);
    results.healthCheck = false;
  }

  // 2. POST /api/v1/legal/query - Supported Query 1
  try {
    const res2 = await makeRequest('POST', '/api/v1/legal/query', {
      question: "What is cheating under the Bharatiya Nyaya Sanhita?"
    });
    console.log('\n2. POST /api/v1/legal/query (Supported Query 1):');
    console.log('Status Code:', res2.status);
    console.log('Response:', JSON.stringify(res2.data, null, 2));
    results.supportedQuery1 = (res2.status === 200 && res2.data.sources && res2.data.sources.length > 0);
  } catch (e) {
    console.error('Supported Query 1 failed:', e.message);
    results.supportedQuery1 = false;
  }

  // 3. POST /api/v1/legal/query - Supported Query 2
  try {
    const res3 = await makeRequest('POST', '/api/v1/legal/query', {
      question: "What is the punishment for murder?"
    });
    console.log('\n3. POST /api/v1/legal/query (Supported Query 2):');
    console.log('Status Code:', res3.status);
    console.log('Response:', JSON.stringify(res3.data, null, 2));
    results.supportedQuery2 = (res3.status === 200 && res3.data.sources && res3.data.sources.length > 0);
  } catch (e) {
    console.error('Supported Query 2 failed:', e.message);
    results.supportedQuery2 = false;
  }

  // 4. POST /api/v1/legal/query - Unsupported Query
  try {
    const res4 = await makeRequest('POST', '/api/v1/legal/query', {
      question: "What is the limitation period for filing a civil property claim?"
    });
    console.log('\n4. POST /api/v1/legal/query (Unsupported Query):');
    console.log('Status Code:', res4.status);
    console.log('Response:', JSON.stringify(res4.data, null, 2));
    results.unsupportedQuery = (res4.status === 200 && res4.data.sources && res4.data.sources.length === 0);
  } catch (e) {
    console.error('Unsupported Query failed:', e.message);
    results.unsupportedQuery = false;
  }

  // 5. POST /api/v1/legal/query - Invalid Input
  try {
    const res5 = await makeRequest('POST', '/api/v1/legal/query', {
      question: "   "
    });
    console.log('\n5. POST /api/v1/legal/query (Invalid Input):');
    console.log('Status Code:', res5.status);
    console.log('Response:', res5.data);
    results.invalidInput = (res5.status === 400);
  } catch (e) {
    console.error('Invalid Input test failed:', e.message);
    results.invalidInput = false;
  }

  // 6. Test AI Service Unavailable behavior
  try {
    const aiService = require('./src/services/aiService');
    const originalConfig = require('./src/config/env');
    // Temporarily override target URL to unassigned port 9999
    const oldUrl = originalConfig.aiServiceUrl;
    originalConfig.aiServiceUrl = 'http://127.0.0.1:9999';

    try {
      await aiService.queryLegalAI("What is theft?");
      results.aiUnavailableHandled = false;
    } catch (err) {
      console.log('\n6. AI Service Unavailable Behavior Test:');
      console.log('Caught Error:', err.name, '| Status:', err.statusCode, '| Message:', err.message);
      results.aiUnavailableHandled = (err.statusCode === 503 && err.message.includes('unavailable'));
    } finally {
      originalConfig.aiServiceUrl = oldUrl;
    }
  } catch (e) {
    console.error('AI Unavailable test failed:', e.message);
    results.aiUnavailableHandled = false;
  }

  console.log("\n--- STAGE 9 BACKEND TEST SUMMARY ---");
  console.log("Health Check Passed:         ", results.healthCheck);
  console.log("Supported Query 1 Passed:    ", results.supportedQuery1);
  console.log("Supported Query 2 Passed:    ", results.supportedQuery2);
  console.log("Unsupported Query Handled:   ", results.unsupportedQuery);
  console.log("Invalid Input Handled (400): ", results.invalidInput);
  console.log("AI Unavailable Handled (503):", results.aiUnavailableHandled);
  console.log("==================================================");
}

if (require.main === module) {
  runTests();
}
