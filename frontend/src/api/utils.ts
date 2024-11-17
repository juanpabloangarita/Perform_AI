import { getOrFetchCSRFCookie } from "../utils";

interface requestArguments {
    url: string;
    method: "POST" | "PUT" | "PATCH" | "DELETE" | "GET";
    body?: string | FormData;
}

export interface TypedResponse<T = object> extends Response {
    json(): Promise<T>;
}

export const request = async <T>({
    url,
    method,
    body,
}: requestArguments): Promise<TypedResponse<T>> => {
    const csrfToken = await getOrFetchCSRFCookie();
    const headers: Record<string, string> = {
        "X-CSRFToken": csrfToken,
    };
    if (typeof body === "string") {
        headers["Content-Type"] = "application/json";
    }
    return fetch(url, { method, body, headers });
};