const { z } = require('zod');
const aiService = require('../services/aiService');

const legalQuerySchema = z.object({
  question: z.string({
    required_error: 'Question is required.',
    invalid_type_error: 'Question must be a string.'
  })
  .trim()
  .min(1, { message: 'Question cannot be empty or whitespace-only.' })
  .max(1000, { message: 'Question cannot exceed 1000 characters.' })
});

async function handleLegalQuery(req, res, next) {
  try {
    const parseResult = legalQuerySchema.safeParse(req.body);

    if (!parseResult.success) {
      const firstIssue = parseResult.error.issues[0];
      return res.status(400).json({
        error: 'Invalid input format',
        message: firstIssue.message
      });
    }

    const { question } = parseResult.data;
    const aiResponse = await aiService.queryLegalAI(question);

    return res.status(200).json(aiResponse);
  } catch (error) {
    next(error);
  }
}

module.exports = {
  handleLegalQuery
};
