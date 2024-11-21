import { ReactNode } from "react";

export const Card = ({
  children,
  title,
}: {
  children: ReactNode;
  title?: string;
}) => (
  <div className="w-full max-w-md p-8 space-y-6 bg-white rounded shadow-md">
    {title && (
      <h1 className="text-2xl font-bold text-center text-gray-700">{title}</h1>
    )}
    {children}
  </div>
);
