"use client";

import { signup } from "@/api/auth";
import { Input } from "@/design-system/Input";
import { Button } from "@/design-system/Button";
import { Card } from "@/design-system/Card";
import Link from "next/link";

export default function SignUpPage() {
  const handleSubmit = async (formData: FormData) => {
    const email = formData.get("email") as string;
    const password = formData.get("password") as string;
    const response = await signup(email, password);
    console.log(response);
  };
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <Card title="Signup">
        <form action={handleSubmit} className="space-y-6">
          <Input label="Email" name="email" type="email" required />
          <Input label="Password" name="password" type="password" required />
          <div>
            <Button type="submit">Sign Up</Button>
          </div>
          <div>
            Already have an account? <Link href="/login">Login</Link>
          </div>
        </form>
      </Card>
    </div>
  );
}
