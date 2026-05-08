/**
 * Main application layout with sidebar navigation and account controls.
 */

import { NavLink, Outlet, useNavigate } from "react-router-dom";
import {
  Activity,
  ChevronDown,
  FileText,
  KeyRound,
  Languages,
  Loader2,
  LogOut,
  Menu,
  Monitor,
  Moon,
  Settings,
  ShieldCheck,
  Sun,
  Upload,
  UserCog,
  Users,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { changePassword } from "../api/client";
import { useAuth } from "../context/AuthContext";
import {
  usePreferences,
  type Language,
  type Theme,
  type TranslationKey,
} from "../context/PreferencesContext";
import Logo from "./Logo";

type NavItem = {
  to: string;
  labelKey: TranslationKey;
  icon: typeof Upload;
  adminOnly?: boolean;
};

const navItems: NavItem[] = [
  { to: "/upload", labelKey: "nav.upload", icon: Upload },
  { to: "/meetings", labelKey: "nav.meetings", icon: FileText },
  { to: "/speakers", labelKey: "nav.speakers", icon: Users },
  { to: "/users", labelKey: "nav.users", icon: UserCog, adminOnly: true },
  { to: "/system", labelKey: "nav.system", icon: Activity },
  { to: "/audit", labelKey: "nav.audit", icon: ShieldCheck, adminOnly: true },
];

const currentYear = new Date().getFullYear();

function Copyright({ className }: { className?: string }) {
  return (
    <p className={`text-[10px] text-epam-gray-400 leading-tight ${className ?? ""}`}>
      &copy; {currentYear} EPAM Law Firm. All rights reserved.
      <br />
      VERITAS — on-premise system. All data stays local.
    </p>
  );
}

export default function Layout() {
  const {
    logout,
    username,
    authRequired,
    role,
    mustChangePassword,
    isAuthenticated,
  } = useAuth();
  const navigate = useNavigate();
  const { language, setLanguage, theme, setTheme, t } = usePreferences();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [passwordOpen, setPasswordOpen] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [passwordSaving, setPasswordSaving] = useState(false);
  const [passwordMessage, setPasswordMessage] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const canManageUsers = role === "admin" || username === "admin" || !authRequired;

  const visibleNavItems = useMemo(
    () =>
      navItems.filter(
        (item) => !item.adminOnly || canManageUsers,
      ),
    [canManageUsers],
  );

  useEffect(() => {
    if (mustChangePassword) {
      setPasswordOpen(true);
      setAccountOpen(false);
    }
  }, [mustChangePassword]);

  function handleLogout() {
    logout();
    navigate("/login");
  }

  async function handlePasswordSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPasswordSaving(true);
    setPasswordError(null);
    setPasswordMessage(null);
    try {
      await changePassword(currentPassword, newPassword);
      setPasswordMessage(t("account.passwordChanged"));
      setCurrentPassword("");
      setNewPassword("");
      window.setTimeout(() => window.location.reload(), 700);
    } catch (err) {
      setPasswordError(
        err instanceof Error ? err.message : t("account.passwordError"),
      );
    } finally {
      setPasswordSaving(false);
    }
  }

  function renderNav(closeMobile = false) {
    return visibleNavItems.map(({ to, labelKey, icon: Icon }) => (
      <NavLink
        key={to}
        to={to}
        onClick={() => closeMobile && setMobileMenuOpen(false)}
        className={({ isActive }) =>
          `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
            isActive
              ? "bg-epam-red text-white"
              : "text-epam-gray-600 hover:bg-epam-gray-100 hover:text-epam-gray-900"
          }`
        }
      >
        <Icon size={18} />
        {t(labelKey)}
      </NavLink>
    ));
  }

  function PreferenceButtons<T extends string>({
    value,
    options,
    onChange,
  }: {
    value: T;
    options: { value: T; label: string; icon?: typeof Sun }[];
    onChange: (value: T) => void;
  }) {
    return (
      <div className="grid grid-cols-2 gap-2">
        {options.map((option) => {
          const Icon = option.icon;
          return (
            <button
              key={option.value}
              onClick={() => onChange(option.value)}
              className={`flex items-center justify-center gap-2 rounded-md border px-2 py-1.5 text-xs font-medium ${
                value === option.value
                  ? "border-epam-red bg-epam-red text-white"
                  : "border-epam-gray-200 text-epam-gray-600 hover:bg-epam-gray-100"
              }`}
            >
              {Icon && <Icon size={14} />}
              {option.label}
            </button>
          );
        })}
      </div>
    );
  }

  const accountPanel = (
    <div className="rounded-lg border border-epam-gray-200 bg-white p-3 shadow-lg">
      <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-epam-gray-900">
        <Settings size={16} className="text-epam-red" />
        {t("account.title")}
      </div>
      <div className="space-y-3">
        <div>
          <div className="mb-1 flex items-center gap-2 text-xs font-medium text-epam-gray-500">
            <Languages size={14} />
            {t("account.language")}
          </div>
          <PreferenceButtons<Language>
            value={language}
            onChange={setLanguage}
            options={[
              { value: "ru", label: "RU" },
              { value: "en", label: "EN" },
            ]}
          />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-epam-gray-500">
            {t("account.theme")}
          </div>
          <PreferenceButtons<Theme>
            value={theme}
            onChange={setTheme}
            options={[
              { value: "light", label: t("account.light"), icon: Sun },
              { value: "dark", label: t("account.dark"), icon: Moon },
              { value: "system", label: t("account.system"), icon: Monitor },
            ]}
          />
        </div>
        {canManageUsers && (
          <button
            onClick={() => {
              setAccountOpen(false);
              setMobileMenuOpen(false);
              navigate("/users");
            }}
            className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-sm font-medium text-epam-gray-700 hover:bg-epam-gray-100"
          >
            <UserCog size={16} />
            {t("account.manageUsers")}
          </button>
        )}
        <button
          onClick={() => {
            setPasswordOpen(true);
            setAccountOpen(false);
          }}
          className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-sm font-medium text-epam-gray-700 hover:bg-epam-gray-100"
        >
          <KeyRound size={16} />
          {t("account.changePassword")}
        </button>
        {isAuthenticated && (
          <button
            onClick={handleLogout}
            className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-sm font-medium text-epam-gray-700 hover:bg-epam-gray-100"
          >
            <LogOut size={16} />
            {t("account.logout")}
          </button>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex h-screen overflow-hidden bg-epam-gray-50 text-epam-gray-900">
      <aside className="hidden md:flex flex-col w-60 bg-white border-r border-epam-gray-200 shrink-0">
        <button
          className="px-6 py-5 border-b border-epam-gray-200 text-left"
          onClick={() => navigate("/meetings")}
        >
          <Logo size="sm" showSubtitle />
        </button>

        <nav className="flex-1 py-4 px-3 space-y-1">{renderNav()}</nav>

        <div className="relative px-3 py-4 border-t border-epam-gray-200">
          <button
            onClick={() => setAccountOpen((open) => !open)}
            className="flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2.5 text-sm font-medium text-epam-gray-700 hover:bg-epam-gray-100"
          >
            <span className="truncate">{username || "VERITAS"}</span>
            <ChevronDown
              size={16}
              className={accountOpen ? "rotate-180 transition-transform" : "transition-transform"}
            />
          </button>
          {accountOpen && (
            <div className="absolute bottom-full left-3 right-3 mb-2">
              {accountPanel}
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-epam-gray-100">
          <Copyright />
        </div>
      </aside>

      <div className="md:hidden fixed top-0 left-0 right-0 z-50 bg-white border-b border-epam-gray-200 px-4 py-3 flex items-center justify-between">
        <Logo size="sm" />
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-2 text-epam-gray-600"
          aria-label="Menu"
        >
          {mobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {mobileMenuOpen && (
        <div className="md:hidden fixed inset-0 z-40 bg-black/30">
          <div className="absolute right-0 top-14 flex h-[calc(100%-3.5rem)] w-72 flex-col bg-white border-l border-epam-gray-200 py-4 px-3 shadow-lg">
            <nav className="space-y-1">{renderNav(true)}</nav>
            <div className="mt-auto pt-4">{accountPanel}</div>
          </div>
        </div>
      )}

      <main className="flex-1 overflow-y-auto md:pt-0 pt-14 flex flex-col">
        <div className="flex-1">
          <Outlet />
        </div>
        <footer className="py-3 px-6 text-center border-t border-epam-gray-100 md:hidden">
          <Copyright className="text-center" />
        </footer>
      </main>

      {passwordOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-4">
          <form
            onSubmit={handlePasswordSubmit}
            className="w-full max-w-md rounded-lg border border-epam-gray-200 bg-white p-5 shadow-xl"
          >
            <div className="mb-4 flex items-center gap-2 text-lg font-semibold text-epam-gray-900">
              <KeyRound size={20} className="text-epam-red" />
              {t("account.changePassword")}
            </div>
            {mustChangePassword && (
              <p className="mb-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                {t("account.mustChange")}
              </p>
            )}
            {(passwordMessage || passwordError) && (
              <p
                className={`mb-4 rounded-md border px-3 py-2 text-sm ${
                  passwordError
                    ? "border-red-200 bg-red-50 text-red-700"
                    : "border-green-200 bg-green-50 text-green-700"
                }`}
              >
                {passwordError || passwordMessage}
              </p>
            )}
            <label className="mb-3 block">
              <span className="mb-1 block text-xs font-medium text-epam-gray-500">
                {t("account.currentPassword")}
              </span>
              <input
                type="password"
                value={currentPassword}
                onChange={(event) => setCurrentPassword(event.target.value)}
                className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
                autoFocus
              />
            </label>
            <label className="mb-5 block">
              <span className="mb-1 block text-xs font-medium text-epam-gray-500">
                {t("account.newPassword")}
              </span>
              <input
                type="password"
                value={newPassword}
                onChange={(event) => setNewPassword(event.target.value)}
                className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
              />
            </label>
            <div className="flex justify-end gap-2">
              {!mustChangePassword && (
                <button
                  type="button"
                  onClick={() => setPasswordOpen(false)}
                  className="rounded-md border border-epam-gray-300 px-4 py-2 text-sm font-medium text-epam-gray-700 hover:bg-epam-gray-100"
                >
                  {t("account.cancel")}
                </button>
              )}
              <button
                type="submit"
                disabled={passwordSaving || !currentPassword || !newPassword}
                className="inline-flex items-center gap-2 rounded-md bg-epam-red px-4 py-2 text-sm font-semibold text-white hover:bg-epam-red-dark disabled:cursor-not-allowed disabled:opacity-50"
              >
                {passwordSaving && <Loader2 size={16} className="animate-spin" />}
                {t("account.savePassword")}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
