const express = require('express');
const cors = require('cors');
const config = require('./config/env');
const healthRoutes = require('./routes/healthRoutes');
const legalRoutes = require('./routes/legalRoutes');
const errorHandler = require('./middleware/errorHandler');

const app = express();

// Enable CORS using configured frontend URL
app.use(cors({
  origin: config.frontendUrl,
  credentials: true,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Body Parser Middleware
app.use(express.json({ limit: '1mb' }));

// Mount API Routes
app.use('/api', healthRoutes);
app.use('/api/v1', legalRoutes);

// Centralized Error Handling Middleware
app.use(errorHandler);

// Start Server if called directly
if (require.main === module) {
  app.listen(config.port, () => {
    console.log(`==================================================`);
    console.log(`  SalahSetu Node.js Backend Foundation Running  `);
    console.log(`==================================================`);
    console.log(`Port:            ${config.port}`);
    console.log(`Environment:     ${config.nodeEnv}`);
    console.log(`CORS Origin:     ${config.frontendUrl}`);
    console.log(`AI Service URL:  ${config.aiServiceUrl}`);
    console.log(`--------------------------------------------------`);
  });
}

module.exports = app;
