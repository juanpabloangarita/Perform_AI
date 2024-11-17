import { FormEvent, useState } from "react";
import {login} from "@/api/auth";

const Login = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    
    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        try {
        await login(email, password);
        } catch (error) {
        console.error(error);
        }
    };
    
    return (
        <form onSubmit={handleSubmit}>
        <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
        />
        <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
        />
        <button type="submit">Login</button>
        </form>
    );
    }

export default Login;