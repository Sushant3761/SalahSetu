const { AIServiceError } = require('../services/aiService');

function errorHandler(err, req, res, next) {
  // Log error internally for debugging without exposing secrets to client
  console.error(`[Error] ${req.method} ${req.url}:`, err.message || err);

  if (err instanceof AIServiceError) {
    return res.status(err.statusCode).json({
      error: 'AI service error',
      message: err.message
    });
  }

  if (err.type === 'entity.parse.failed') {
    return res.status(400).json({
      error: 'Invalid JSON',
      message: 'The request payload contains invalid JSON.'
    });
  }

  return res.status(500).json({
    error: 'Internal server error',
    message: 'An unexpected error occurred. Please try again later.'
  });
}

module.exports = errorHandler;
