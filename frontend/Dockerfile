# Frontend Dockerfile — multi-stage build (Node build -> Nginx serve)
#
# Build:  docker build -t crop-disease-frontend \
#           --build-arg REACT_APP_API_BASE_URL=http://localhost:8000 ./frontend
# Run:    docker run -p 3000:80 crop-disease-frontend
# (Normally run via docker-compose — see docker-compose.yml at repo root)
#
# NOTE ON REACT_APP_API_BASE_URL: Create React App bakes REACT_APP_*
# environment variables into the JS bundle at BUILD time, not at
# container-run time. This means the backend URL must be known when
# `docker build` runs. docker-compose.yml passes this automatically for
# local development. For a production deployment where the frontend is
# served from a different host than assumed at build time, rebuild the
# image with the correct --build-arg, or serve the frontend through a
# reverse proxy that maps the same origin (see docs/deployment.md).

# --- Stage 1: build the static React app ---
FROM node:18-alpine AS build
WORKDIR /app

COPY package.json ./
RUN npm install --silent

COPY . .

ARG REACT_APP_API_BASE_URL=http://localhost:8000
ENV REACT_APP_API_BASE_URL=${REACT_APP_API_BASE_URL}

RUN npm run build

# --- Stage 2: serve with Nginx ---
FROM nginx:1.27-alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
