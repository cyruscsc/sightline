FROM --platform=linux/amd64 node:20-alpine AS builder
# FROM node:20-alpine as builder

WORKDIR /app

COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

COPY . .
RUN yarn build


FROM --platform=linux/amd64 node:20-alpine
# FROM node:20-alpine

WORKDIR /app

COPY package.json yarn.lock ./
RUN yarn install --production --frozen-lockfile

COPY --from=builder /app/build ./build
COPY --from=builder /app/package.json ./

EXPOSE 3000

ENV NODE_ENV=production
CMD ["node", "build"]