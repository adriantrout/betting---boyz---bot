# Betting Boyz — GitHub-safe runner (NO SECRETS IN FILE)
import os
import sys
from datetime import datetime, timezone

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

import betting_boyz_bot as bot

if __name__ == "__main__":
    print("Betting Boyz Runner Started", datetime.now(timezone.utc).isoformat())
    args = bot.argparse.Namespace(
        refresh_sports=False,
        self_test=False,
        slot=None,
    )
    bot.main(args)
