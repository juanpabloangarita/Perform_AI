import { request } from "@/api/fetcher";

export const login = (email: string, password: string) => {
  return request({
    url: "/api/auth/login/",
    method: "POST",
    body: { email, password },
  });
};

export const signup = (email: string, password: string) => {
  return request({
    url: "/api/auth/signup/",
    method: "POST",
    body: { email, password },
  });
};
