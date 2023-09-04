#!/bin/sh
set -e
cargo publish
VERSION=$(cat Cargo.toml | grep ^version | cut -d'"' -f 2)
TAG_NAME="v$VERSION"
echo "creating $TAG_NAME"
git tag -a $TAG_NAME -m "version $VERSION"
git push origin $TAG_NAME