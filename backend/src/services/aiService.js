const config = require('../config/env');

class AIServiceError extends Error {
  constructor(message, statusCode = 503, details = null) {
    super(message);
    this.name = 'AIServiceError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

/**
 * Communicates with Python AI Service REST API.
 */
async function queryLegalAI(question) {
  const targetUrl = `${config.aiServiceUrl}/api/v1/legal/query`;

  try {
    const response = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(config.aiServiceTimeoutMs)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      if (response.status === 400) {
        throw new AIServiceError(errorData.message || 'Invalid request format.', 400);
      }
      throw new AIServiceError('The legal intelligence service encountered an error.', 502);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    if (error.name === 'AIServiceError') {
      throw error;
    }
    if (error.name === 'TimeoutError' || error.name === 'AbortError') {
      throw new AIServiceError('The legal intelligence service timed out.', 504);
    }
    // Connection refused / Network error
    throw new AIServiceError('The legal intelligence service is temporarily unavailable.', 503);
  }
}

module.exports = {
  queryLegalAI,
  AIServiceError
};
