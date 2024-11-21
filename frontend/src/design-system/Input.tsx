export const Input = ({
  type = "text",
  label,
  name,
  required = false,
}: {
  type?: "text" | "email" | "password";
  label: string;
  name: string;
  required?: boolean;
}) => (
  <div>
    <label htmlFor={name} className="block text-sm font-medium text-gray-700">
      {label}
    </label>
    <input
      type={type}
      id={name}
      name={name}
      required={required}
      className="w-full px-3 py-2 mt-1 text-gray-500 border rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
    />
  </div>
);
