FROM node:22-bookworm-slim

WORKDIR /app

ENV NODE_ENV=development \
    EPAM_FRONTEND_PORT=5173 \
    EPAM_BACKEND_HOST=backend \
    EPAM_BACKEND_PORT=8000

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
