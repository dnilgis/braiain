#!/usr/bin/env bash
# Push the commit this run just made, without ever downloading the repo's history.
#
# The workflow checks out only the newest commit (depth 1). "git pull --rebase" on a shallow
# copy can end up fetching thousands of old objects, which hung runs for 20 minutes. Instead:
#   1. try a plain push (works whenever nobody pushed since this run started);
#   2. otherwise fetch only the newest commit, reset to it, put back exactly the files this
#      run's commit changed, commit again with the same message, and push.
# Only bot-written files are in the run's commit (data, models, v, daily), so step 2 cannot
# undo a person's edit to anything else.
set -u
export GIT_TERMINAL_PROMPT=0
msg=$(git log -1 --pretty=%B)
files=$(git diff --name-only HEAD~1 HEAD)
[ -z "$files" ] && exit 0
tmp=$(mktemp -d)
while IFS= read -r f; do
  if [ -e "$f" ]; then mkdir -p "$tmp/$(dirname "$f")"; cp -p "$f" "$tmp/$f"; fi
done <<< "$files"

for i in 1 2 3 4; do
  if timeout 120 git push -q origin HEAD:main; then echo "pushed (try $i)"; exit 0; fi
  echo "push try $i rejected or timed out; replaying this run's files on the newest commit"
  if ! timeout 120 git fetch -q --depth=1 origin main; then sleep 10; continue; fi
  git reset -q --hard FETCH_HEAD
  while IFS= read -r f; do
    if [ -e "$tmp/$f" ]; then mkdir -p "$(dirname "$f")"; cp -p "$tmp/$f" "$f"; git add -- "$f"
    else git rm -q --ignore-unmatch -- "$f"; fi
  done <<< "$files"
  if git diff --cached --quiet; then echo "nothing left to push"; exit 0; fi
  git commit -q -m "$msg"
done
echo "could not push after 4 tries" >&2
exit 1
