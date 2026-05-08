/**
 * Login page with JWT authentication.
 *
 * If auth is not yet enforced by the backend (Block 2 in progress),
 * allows the user to proceed without credentials.
 */

import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Lock, AlertCircle, ServerOff } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import Logo from "../components/Logo";

export default function LoginPage() {
  const { login, skipAuth, backendReachable, authRequired } = useAuth();
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await login(username, password);
      navigate("/upload");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setLoading(false);
    }
  }

  function handleSkipAuth() {
    skipAuth();
    navigate("/upload");
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-epam-gray-50 px-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-10">
          <Logo size="lg" showSubtitle />
        </div>

        {/* Backend status warning */}
        {!backendReachable && (
          <div className="mb-6 flex items-start gap-3 p-4 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">
            <ServerOff size={18} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">Backend is not reachable</p>
              <p className="mt-1 text-amber-600">
                Ensure the VERITAS backend is running. Default port is 8765;
                check the cmd window labeled "VERITAS Backend".
              </p>
            </div>
          </div>
        )}

        {/* Login form */}
        <div className="bg-white rounded-xl shadow-sm border border-epam-gray-200 p-8">
          <div className="flex items-center gap-2 mb-6">
            <Lock size={20} className="text-epam-red" />
            <h2 className="font-heading text-xl text-epam-gray-900">
              Sign in
            </h2>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="username"
                className="block text-sm font-medium text-epam-gray-700 mb-1"
              >
                Username
              </label>
              <input
                id="username"
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-3 py-2 border border-epam-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red transition-colors"
                placeholder="Enter your username"
                autoComplete="username"
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-epam-gray-700 mb-1"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-3 py-2 border border-epam-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red transition-colors"
                placeholder="Enter your password"
                autoComplete="current-password"
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 text-sm text-red-600">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Signing in..." : "Sign in"}
            </button>
          </form>

          {/* Skip auth when backend doesn't enforce it yet */}
          {!authRequired && (
            <div className="mt-4 pt-4 border-t border-epam-gray-100">
              <button
                onClick={handleSkipAuth}
                className="w-full py-2 text-sm text-epam-gray-500 hover:text-epam-gray-700 transition-colors"
              >
                Continue without authentication
              </button>
              <p className="mt-2 text-xs text-epam-gray-400 text-center">
                Auth endpoint not detected. Will be enforced after Block 2.
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="mt-6 text-center text-[10px] text-epam-gray-400 leading-tight">
          &copy; {new Date().getFullYear()} EPAM Law Firm. All rights reserved.
          <br />
          VERITAS — on-premise system. All data stays local.
        </p>
      </div>
    </div>
  );
}
