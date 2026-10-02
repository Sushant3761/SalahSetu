const express = require('express');
const router = express.Router();
const legalController = require('../controllers/legalController');

router.post('/legal/query', legalController.handleLegalQuery);

module.exports = router;
