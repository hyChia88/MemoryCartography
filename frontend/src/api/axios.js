// src/api/axios.js
import axios from 'axios';

const BASE_URL = 'http://localhost:8000';  // 注意没有 /api

const api = axios.create({
  baseURL: BASE_URL,
  withCredentials: true  // 如果需要跨域携带 cookies
});

export default api;