// frontend/src/utils/logger.js
// Production-ready logging utility

const LOG_LEVELS = {
  ERROR: 'ERROR',
  WARN: 'WARN',
  INFO: 'INFO',
  DEBUG: 'DEBUG'
};

class Logger {
  constructor() {
    this.isDevelopment = process.env.NODE_ENV === 'development';
    this.isProduction = process.env.NODE_ENV === 'production';
  }

  formatMessage(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      message,
      ...(data && { data })
    };

    if (this.isDevelopment) {
      // In development, use console with colors
      const styles = {
        ERROR: 'color: red; font-weight: bold',
        WARN: 'color: orange; font-weight: bold',
        INFO: 'color: blue',
        DEBUG: 'color: gray'
      };
      console.log(`%c[${level}]`, styles[level] || '', message, data || '');
    } else if (this.isProduction) {
      // In production, log to console as JSON for log aggregation services
      console.log(JSON.stringify(logEntry));
    }

    return logEntry;
  }

  error(message, error = null) {
    const errorData = error ? {
      message: error.message,
      stack: error.stack,
      name: error.name
    } : null;
    return this.formatMessage(LOG_LEVELS.ERROR, message, errorData);
  }

  warn(message, data = null) {
    return this.formatMessage(LOG_LEVELS.WARN, message, data);
  }

  info(message, data = null) {
    return this.formatMessage(LOG_LEVELS.INFO, message, data);
  }

  debug(message, data = null) {
    if (this.isDevelopment) {
      return this.formatMessage(LOG_LEVELS.DEBUG, message, data);
    }
    return null;
  }

  // API specific logging
  apiCall(endpoint, method = 'GET', data = null) {
    if (this.isDevelopment) {
      this.debug(`API Call: ${method} ${endpoint}`, data);
    }
  }

  apiSuccess(endpoint, response = null) {
    if (this.isDevelopment) {
      this.debug(`API Success: ${endpoint}`, response);
    }
  }

  apiError(endpoint, error = null) {
    this.error(`API Error: ${endpoint}`, error);
  }
}

// Export singleton instance
export const logger = new Logger();
export default logger;
