#!/bin/sh
set -e
rm -rf /etc/nginx/sites-enabled
rm -f /etc/nginx/sites-available/default
exec /usr/sbin/nginx -g "daemon off;"
