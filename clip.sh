#!/usr/bin/env bash
ACTION=$1
CHANNEL=$2
OUT_DIR="$(cd "$(dirname "$0")" && pwd)/output"

case "$ACTION" in
  add)
    [ -z "$CHANNEL" ] && echo "Usage: ./clip.sh add <chaine>" && exit 1
    docker run -d --name "clip-$CHANNEL" --restart unless-stopped \
      -v "$OUT_DIR:/app/output" \
      twitch-clipper "$CHANNEL" --out /app/output --model base --device cpu
    echo "Surveillance lancée pour $CHANNEL"
    ;;
  rm)
    [ -z "$CHANNEL" ] && echo "Usage: ./clip.sh rm <chaine>" && exit 1
    docker rm -f "clip-$CHANNEL"
    echo "Surveillance arrêtée pour $CHANNEL"
    ;;
  list)
    docker ps --filter "name=clip-" --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}"
    ;;
  logs)
    [ -z "$CHANNEL" ] && echo "Usage: ./clip.sh logs <chaine>" && exit 1
    docker logs -f --tail 30 "clip-$CHANNEL"
    ;;
  *)
    echo "Commandes : ./clip.sh {add|rm|list|logs} [chaine]"
    ;;
esac    