import { request } from "./utils";

export const login = async (email: string, password: string) => {
    const response = await request({
        url: `/api/auth/login/`, 
        method: 'POST',
        body: JSON.stringify({ email, password }),
    });
    
    if (!response.ok) {
        throw new Error('Invalid credentials');
    }
    
    const { accessToken } = await response.json();
    localStorage.setItem('accessToken', accessToken);
    }