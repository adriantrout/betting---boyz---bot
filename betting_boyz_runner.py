# Betting Boyz — GitHub-safe runner
import os
import sys
from datetime import datetime

REQUIRED_VARS = [
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_WHATSAPP_FROM",
    "TO_WHATSAPP_NUMBER",
    "ODDS_API_KEY",
    "WHATSAPP_CHANNEL_LINK",
]

missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    print("Missing environment variables:", missing)
    sys.exit(1)

import betting_boyz_real_final_verified_FULL_PATCHED as bot

if __name__ == "__main__":
    print("Betting Boyz Runner Started", datetime.utcnow())
    bot.main(bot.argparse.Namespace(refresh_sports=False, self_test=False))
