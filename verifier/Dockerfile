# use the official Bun image
# see all versions at https://hub.docker.com/r/oven/bun/tags
FROM node:22 AS base
WORKDIR /usr/src/app

# install cmake curl python3  cmake, ninja, and the vcpkg dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc cmake zip unzip curl python3 ninja-build \
  libzmq3-dev

# install dependencies into temp directory
# this will cache them and speed up future builds
FROM base AS install
# install with --production (exclude devDependencies)
RUN mkdir -p /temp/prod
COPY package.json /temp/prod/
RUN --mount=type=cache,target=/temp/prod/.npm \
  cd /temp/prod && \
  npm set cache /temp/prod/.npm && \
  VCPKG_FORCE_SYSTEM_BINARIES=1 npm install --frozen-lockfile --omit=dev

# copy node_modules from temp directory
# then copy all (non-ignored) project files into the image
FROM base AS prerelease
COPY . .

# copy production dependencies and source code into final image
FROM base AS release
COPY --from=install /temp/prod/node_modules node_modules
COPY --from=prerelease /usr/src/app/index.ts .
COPY --from=prerelease /usr/src/app/package.json .

# run the app
EXPOSE 3000/tcp
ENTRYPOINT [ "node", "--experimental-strip-types", "index.ts" ]
