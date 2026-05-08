import { useCallback, useEffect, useState } from "react";
import {
  Check,
  KeyRound,
  Loader2,
  RefreshCw,
  ShieldAlert,
  Trash2,
  UserPlus,
} from "lucide-react";
import {
  createUserAccount,
  deleteUserAccount,
  listUsers,
  resetUserPassword,
  updateUserAccount,
} from "../api/client";
import type { UserAccount } from "../types/api";
import { useAuth } from "../context/AuthContext";
import { usePreferences } from "../context/PreferencesContext";

const roles = ["admin", "operator", "lawyer", "viewer"];

export default function UsersPage() {
  const { role, username, authRequired } = useAuth();
  const { t } = usePreferences();
  const [users, setUsers] = useState<UserAccount[]>([]);
  const [loading, setLoading] = useState(true);
  const [savingUser, setSavingUser] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newUser, setNewUser] = useState({
    username: "",
    password: "",
    role: "operator",
    must_change_password: true,
  });
  const [resetPasswords, setResetPasswords] = useState<Record<string, string>>({});

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setUsers(await listUsers());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load users");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function createUser() {
    setError(null);
    setMessage(null);
    setSavingUser("new");
    try {
      const created = await createUserAccount(newUser);
      setUsers((current) => [...current, created].sort((a, b) => a.username.localeCompare(b.username)));
      setNewUser({
        username: "",
        password: "",
        role: "operator",
        must_change_password: true,
      });
      setMessage(t("users.created"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create user");
    } finally {
      setSavingUser(null);
    }
  }

  async function updateUser(username: string, patch: Partial<UserAccount>) {
    setError(null);
    setMessage(null);
    setSavingUser(username);
    try {
      const updated = await updateUserAccount(username, patch);
      setUsers((current) =>
        current.map((user) => (user.username === username ? updated : user)),
      );
      setMessage(t("users.updated"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update user");
    } finally {
      setSavingUser(null);
    }
  }

  async function resetPassword(username: string) {
    const password = resetPasswords[username] || "";
    if (!password) return;
    setError(null);
    setMessage(null);
    setSavingUser(`${username}:password`);
    try {
      const updated = await resetUserPassword(username, password);
      setUsers((current) =>
        current.map((user) => (user.username === username ? updated : user)),
      );
      setResetPasswords((current) => ({ ...current, [username]: "" }));
      setMessage(t("users.passwordReset"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset password");
    } finally {
      setSavingUser(null);
    }
  }

  async function deleteUser(username: string) {
    if (!window.confirm(`Delete user ${username}?`)) return;
    setError(null);
    setMessage(null);
    setSavingUser(`${username}:delete`);
    try {
      await deleteUserAccount(username);
      setUsers((current) => current.filter((user) => user.username !== username));
      setMessage(t("users.deleted"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete user");
    } finally {
      setSavingUser(null);
    }
  }

  const canManageUsers = role === "admin" || username === "admin" || !authRequired;

  if (!canManageUsers) {
    return (
      <div className="p-6 max-w-3xl">
        <div className="flex items-center gap-3 rounded-lg border border-epam-gray-200 bg-white p-4 text-epam-gray-700">
          <ShieldAlert size={20} className="text-epam-red" />
          Admin access required.
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-epam-gray-900">
            {t("users.title")}
          </h1>
          <p className="text-sm text-epam-gray-500 mt-1">
            {t("users.subtitle")}
          </p>
        </div>
        <button
          onClick={load}
          className="inline-flex items-center gap-2 rounded-md border border-epam-gray-300 px-3 py-2 text-sm font-medium text-epam-gray-700 hover:bg-epam-gray-100"
        >
          <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          {t("users.refresh")}
        </button>
      </div>

      {(message || error) && (
        <div
          className={`rounded-md border px-4 py-3 text-sm ${
            error
              ? "border-red-200 bg-red-50 text-red-700"
              : "border-green-200 bg-green-50 text-green-700"
          }`}
        >
          {error || message}
        </div>
      )}

      <section className="rounded-lg border border-epam-gray-200 bg-white">
        <div className="border-b border-epam-gray-200 px-4 py-3">
          <h2 className="text-base font-semibold text-epam-gray-900">
            {t("users.create")}
          </h2>
        </div>
        <div className="grid gap-3 p-4 md:grid-cols-[1fr_1fr_160px_auto_auto] md:items-end">
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-epam-gray-500">
              {t("users.username")}
            </span>
            <input
              value={newUser.username}
              onChange={(event) =>
                setNewUser((current) => ({
                  ...current,
                  username: event.target.value,
                }))
              }
              className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-epam-gray-500">
              {t("users.password")}
            </span>
            <input
              type="password"
              value={newUser.password}
              onChange={(event) =>
                setNewUser((current) => ({
                  ...current,
                  password: event.target.value,
                }))
              }
              className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-epam-gray-500">
              {t("users.role")}
            </span>
            <select
              value={newUser.role}
              onChange={(event) =>
                setNewUser((current) => ({
                  ...current,
                  role: event.target.value,
                }))
              }
              className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
            >
              {roles.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 rounded-md border border-epam-gray-200 px-3 py-2 text-sm text-epam-gray-700">
            <input
              type="checkbox"
              checked={newUser.must_change_password}
              onChange={(event) =>
                setNewUser((current) => ({
                  ...current,
                  must_change_password: event.target.checked,
                }))
              }
              className="accent-epam-red"
            />
            {t("users.mustChange")}
          </label>
          <button
            onClick={createUser}
            disabled={
              savingUser === "new" || !newUser.username || !newUser.password
            }
            className="inline-flex items-center justify-center gap-2 rounded-md bg-epam-red px-4 py-2 text-sm font-semibold text-white hover:bg-epam-red-dark disabled:cursor-not-allowed disabled:opacity-50"
          >
            {savingUser === "new" ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <UserPlus size={16} />
            )}
            {t("users.create")}
          </button>
        </div>
      </section>

      <section className="overflow-hidden rounded-lg border border-epam-gray-200 bg-white">
        {loading ? (
          <div className="flex items-center justify-center py-12 text-epam-gray-500">
            <Loader2 size={22} className="animate-spin" />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-epam-gray-200">
              <thead className="bg-epam-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-epam-gray-500">
                    {t("users.username")}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-epam-gray-500">
                    {t("users.role")}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-epam-gray-500">
                    {t("users.active")}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-epam-gray-500">
                    {t("users.mustChange")}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-epam-gray-500">
                    {t("users.reset")}
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-epam-gray-500">
                    {t("users.delete")}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-epam-gray-100">
                {users.map((user) => (
                  <tr key={user.username}>
                    <td className="whitespace-nowrap px-4 py-3 text-sm font-medium text-epam-gray-900">
                      {user.username}
                    </td>
                    <td className="px-4 py-3">
                      <select
                        value={user.role}
                        onChange={(event) =>
                          updateUser(user.username, { role: event.target.value })
                        }
                        className="rounded-md border border-epam-gray-300 bg-white px-2 py-1.5 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
                      >
                        {roles.map((item) => (
                          <option key={item} value={item}>
                            {item}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() =>
                          updateUser(user.username, {
                            is_active: !user.is_active,
                          })
                        }
                        className={`inline-flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-sm ${
                          user.is_active
                            ? "border-green-200 bg-green-50 text-green-700"
                            : "border-epam-gray-200 bg-epam-gray-50 text-epam-gray-500"
                        }`}
                      >
                        {savingUser === user.username ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : user.is_active ? (
                          <Check size={14} />
                        ) : null}
                        {user.is_active ? "Yes" : "No"}
                      </button>
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() =>
                          updateUser(user.username, {
                            must_change_password: !user.must_change_password,
                          })
                        }
                        className={`rounded-md border px-2.5 py-1.5 text-sm ${
                          user.must_change_password
                            ? "border-amber-200 bg-amber-50 text-amber-700"
                            : "border-epam-gray-200 bg-epam-gray-50 text-epam-gray-500"
                        }`}
                      >
                        {user.must_change_password ? "Yes" : "No"}
                      </button>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex min-w-[260px] items-center gap-2">
                        <input
                          type="password"
                          value={resetPasswords[user.username] || ""}
                          onChange={(event) =>
                            setResetPasswords((current) => ({
                              ...current,
                              [user.username]: event.target.value,
                            }))
                          }
                          className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-1.5 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
                        />
                        <button
                          onClick={() => resetPassword(user.username)}
                          disabled={
                            savingUser === `${user.username}:password` ||
                            !resetPasswords[user.username]
                          }
                          className="inline-flex items-center gap-1.5 rounded-md border border-epam-gray-300 px-2.5 py-1.5 text-sm font-medium text-epam-gray-700 hover:bg-epam-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          {savingUser === `${user.username}:password` ? (
                            <Loader2 size={14} className="animate-spin" />
                          ) : (
                            <KeyRound size={14} />
                          )}
                          {t("users.reset")}
                        </button>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => deleteUser(user.username)}
                        disabled={savingUser === `${user.username}:delete`}
                        className="inline-flex items-center gap-1.5 rounded-md border border-red-200 px-2.5 py-1.5 text-sm font-medium text-red-700 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        {savingUser === `${user.username}:delete` ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <Trash2 size={14} />
                        )}
                        {t("users.delete")}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
