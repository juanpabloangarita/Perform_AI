export interface TypedResponse<T> extends Response {
  json(): Promise<T>;
}

export const request = async <T>({
  url,
  method = "GET",
  body,
}: {
  url: string;
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: Record<string, unknown>;
}): Promise<TypedResponse<T>> => {
  const csrfToken = await getOrFetchCSRFCookie();
  return fetch(url, {
    method,
    body: body ? JSON.stringify(body) : undefined,
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfToken,
    },
  });
};

export const getOrFetchCSRFCookie = async (): Promise<string> => {
  try {
    return getCookie("csrftoken");
  } catch {
    // Fetching a random Django page make sure Django set the CSRF token
    await fetch("/api/auth/login/");
    return getCookie("csrftoken");
  }
};

const getCookie = (cookieKey: string) => {
  if (!document.cookie || document.cookie === "") {
    throw Error(`There is no cookie`);
  }
  const cookies = document.cookie.split(";");
  for (let i = 0; i < cookies.length; i++) {
    const cookie = cookies[i].trim();
    if (cookie.substring(0, cookieKey.length + 1) === cookieKey + "=") {
      return decodeURIComponent(cookie.substring(cookieKey.length + 1));
    }
  }
  throw Error(`Cookie ${cookieKey} not found.`);
};
