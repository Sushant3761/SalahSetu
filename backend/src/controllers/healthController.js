function getHealth(req, res) {
  return res.status(200).json({
    status: 'ok',
    service: 'SalahSetu Backend'
  });
}

module.exports = {
  getHealth
};
