'handlers': {
    'file': {
        'class': 'logging.handlers.RotatingFileHandler',
        'filename': 'logs/conversation.log',
        'maxBytes': 5242880,  # 5MB
        'backupCount': 3,
    }
} 