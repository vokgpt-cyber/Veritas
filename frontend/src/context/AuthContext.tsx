/**
 * Authentication context provider.
 *
 * Manages JWT token state and provides login/logout to the app.
 * While Block 2 auth may not be fully implemented yet, the frontend
 * is ready to integrate once the /api/auth/login endpoint exists.
 */

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import {
  login as apiLogin,
  checkHealth,
  getCurrentUser,
  getToken,
  setToken,
  clearToken,
  logoutSession,
} from "../api/client";

interface AuthContextValue {
  isAuthenticated: boolean;
  isLoading: boolean;
  username: string | null;
  role: string | null;
  mustChangePassword: boolean;
  backendReachable: boolean;
  authRequired: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  skipAuth: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [username, setUsername] = useState<string | null>(null);
  const [role, setRole] = useState<string | null>(null);
  const [mustChangePassword, setMustChangePassword] = useState(false);
  const [backendReachable, setBackendReachable] = useState(false);
  const [authRequired, setAuthRequired] = useState(true);

  // On mount, check if we have a token and if backend is up
  useEffect(() => {
    async function init() {
      const reachable = await checkHealth();
      setBackendReachable(reachable);
      const token = getToken();

      if (reachable) {
        // Try to access a protected endpoint to check if auth is enforced
        try {
          const response = await fetch("/api/meetings/", {
            headers: token
              ? { Authorization: `Bearer ${token}` }
              : {},
          });

          if (response.status === 401 || response.status === 403) {
            // Auth is enforced
            setAuthRequired(true);
            if (token) {
              try {
                const me = await getCurrentUser();
                setUsername(me.username);
                setRole(me.role);
                setMustChangePassword(me.must_change_password);
                setIsAuthenticated(true);
              } catch {
                clearToken();
                setIsAuthenticated(false);
              }
            }
          } else if (token) {
            // Token is valid and backend accepted it. Fetch /me so
            // role-sensitive navigation (Users/Audit) is correct after
            // page refreshes and after older tokens.
            try {
              const me = await getCurrentUser();
              setAuthRequired(true);
              setUsername(me.username);
              setRole(me.role);
              setMustChangePassword(me.must_change_password);
              setIsAuthenticated(true);
            } catch {
              clearToken();
              setAuthRequired(false);
              setUsername("admin");
              setRole("admin");
              setMustChangePassword(false);
              setIsAuthenticated(true);
            }
          } else {
            // No auth enforced. Treat local desktop mode as admin so
            // the pilot can still access the user-management screen.
            setAuthRequired(false);
            setUsername("admin");
            setRole("admin");
            setMustChangePassword(false);
            setIsAuthenticated(true);
          }
        } catch {
          // Backend not reachable or error
          setAuthRequired(false);
          setUsername("admin");
          setRole("admin");
          setMustChangePassword(false);
        }
      }

      setIsLoading(false);
    }

    init();
  }, []);

  const login = useCallback(async (user: string, pass: string) => {
    try {
      const auth = await apiLogin(user, pass);
      const me = await getCurrentUser().catch(() => ({
        username: auth.username || user,
        role: auth.role || (user === "admin" ? "admin" : "operator"),
        must_change_password: auth.must_change_password || false,
      }));
      setUsername(me.username);
      setRole(me.role || (me.username === "admin" ? "admin" : "operator"));
      setMustChangePassword(me.must_change_password);
      setIsAuthenticated(true);
    } catch (error) {
      // If auth endpoint doesn't exist yet, allow through
      const msg = error instanceof Error ? error.message : "";
      if (msg.includes("404") || msg.includes("Not Found")) {
        // Auth not implemented yet, set a placeholder token
        setToken("dev-mode-no-auth");
        setUsername(user);
        setRole("admin");
        setMustChangePassword(false);
        setIsAuthenticated(true);
        setAuthRequired(false);
      } else {
        throw error;
      }
    }
  }, []);

  const logout = useCallback(() => {
    logoutSession().catch(() => undefined);
    clearToken();
    setIsAuthenticated(false);
    setUsername(null);
    setRole(null);
    setMustChangePassword(false);
  }, []);

  const skipAuth = useCallback(() => {
    setToken("dev-mode-no-auth");
    setIsAuthenticated(true);
    setUsername("admin");
    setRole("admin");
    setMustChangePassword(false);
    setAuthRequired(false);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        isLoading,
        username,
        role,
        mustChangePassword,
        backendReachable,
        authRequired,
        login,
        logout,
        skipAuth,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
