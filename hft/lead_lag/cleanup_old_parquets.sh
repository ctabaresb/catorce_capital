#!/bin/bash
# cleanup_old_parquets.sh
# Deletes parquet files older than 4 hours from data/ directory.
# By 4 hours, the hourly S3 cron has had 3+ chances to sync them.
# Zero S3 API calls. Just local file deletion.
#
# Add to crontab:
#   crontab -e
#   0 * * * * /home/ec2-user/data_extraction/cleanup_old_parquets.sh >> /home/ec2-user/data_extraction/logs/cleanup.log 2>&1

DATA_DIR="/home/ec2-user/data_extraction/data"
AGE_MINUTES=240  # 4 hours

COUNT=$(find "$DATA_DIR" -name "*.parquet" -mmin +$AGE_MINUTES -type f | wc -l)

if [ "$COUNT" -gt 0 ]; then
    find "$DATA_DIR" -name "*.parquet" -mmin +$AGE_MINUTES -type f -delete
    echo "$(date '+%Y-%m-%d %H:%M:%S') CLEANUP: deleted $COUNT parquet files older than ${AGE_MINUTES}min"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') CLEANUP: nothing to delete"
fi
