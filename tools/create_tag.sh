#!/bin/sh
set -e
cargo publish
VERSION=$(cat Cargo.toml | grep ^version | egrep -i -o '\d*\.\d*(\.\d*)?')
TAG_NAME="v$VERSION"
echo "creating $TAG_NAME"
git tag -a $TAG_NAME -m "version $VERSION"
git push origin $TAG_NAME