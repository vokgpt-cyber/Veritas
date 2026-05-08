import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type Language = "ru" | "en";
export type Theme = "light" | "dark" | "system";
type ResolvedTheme = "light" | "dark";

const LANGUAGE_KEY = "veritas_language";
const THEME_KEY = "veritas_theme";

const messages = {
  en: {
    "nav.upload": "Upload",
    "nav.meetings": "Meetings",
    "nav.speakers": "Speakers",
    "nav.users": "Users",
    "nav.developer": "Developer",
    "nav.system": "System",
    "nav.audit": "Audit",
    "account.title": "Account",
    "account.language": "Interface language",
    "account.theme": "Theme",
    "account.light": "Light",
    "account.dark": "Dark",
    "account.system": "System",
    "account.manageUsers": "Manage users",
    "account.developerSettings": "Developer settings",
    "account.changePassword": "Change password",
    "account.logout": "Log out",
    "account.currentPassword": "Current password",
    "account.newPassword": "New password",
    "account.savePassword": "Save password",
    "account.cancel": "Cancel",
    "account.mustChange": "You need to set a new password before continuing.",
    "account.passwordChanged": "Password changed. Reloading account data.",
    "account.passwordError": "Could not change password.",
    "users.title": "Users",
    "users.subtitle": "Create accounts, assign roles, and reset temporary passwords.",
    "users.create": "Create user",
    "users.username": "Username",
    "users.password": "Temporary password",
    "users.role": "Role",
    "users.active": "Active",
    "users.mustChange": "Must change password",
    "users.reset": "Reset password",
    "users.delete": "Delete",
    "users.save": "Save",
    "users.refresh": "Refresh",
    "users.created": "User created.",
    "users.updated": "User updated.",
    "users.passwordReset": "Password reset.",
    "users.deleted": "User deleted.",
  },
  ru: {
    "nav.upload": "Загрузка",
    "nav.meetings": "Материалы",
    "nav.speakers": "Голоса",
    "nav.users": "Пользователи",
    "nav.developer": "Разработчик",
    "nav.system": "Система",
    "nav.audit": "Аудит",
    "account.title": "Учетная запись",
    "account.language": "Язык интерфейса",
    "account.theme": "Тема",
    "account.light": "Светлая",
    "account.dark": "Темная",
    "account.system": "Как в системе",
    "account.manageUsers": "Управление пользователями",
    "account.developerSettings": "Настройки разработчика",
    "account.changePassword": "Сменить пароль",
    "account.logout": "Выйти",
    "account.currentPassword": "Текущий пароль",
    "account.newPassword": "Новый пароль",
    "account.savePassword": "Сохранить пароль",
    "account.cancel": "Отмена",
    "account.mustChange": "Нужно задать новый пароль перед продолжением.",
    "account.passwordChanged": "Пароль изменен. Обновляю данные учетной записи.",
    "account.passwordError": "Не удалось сменить пароль.",
    "users.title": "Пользователи",
    "users.subtitle": "Создание учеток, роли и временные пароли для пилотной группы.",
    "users.create": "Создать пользователя",
    "users.username": "Логин",
    "users.password": "Временный пароль",
    "users.role": "Роль",
    "users.active": "Активен",
    "users.mustChange": "Сменить пароль при входе",
    "users.reset": "Сбросить пароль",
    "users.delete": "Удалить",
    "users.save": "Сохранить",
    "users.refresh": "Обновить",
    "users.created": "Пользователь создан.",
    "users.updated": "Пользователь обновлен.",
    "users.passwordReset": "Пароль сброшен.",
    "users.deleted": "Пользователь удален.",
  },
} as const;

export type TranslationKey = keyof typeof messages.en;

interface PreferencesContextValue {
  language: Language;
  theme: Theme;
  resolvedTheme: ResolvedTheme;
  setLanguage: (language: Language) => void;
  setTheme: (theme: Theme) => void;
  t: (key: TranslationKey) => string;
}

const PreferencesContext = createContext<PreferencesContextValue | null>(null);

function readLanguage(): Language {
  const stored = localStorage.getItem(LANGUAGE_KEY);
  return stored === "en" || stored === "ru" ? stored : "ru";
}

function readTheme(): Theme {
  const stored = localStorage.getItem(THEME_KEY);
  return stored === "light" || stored === "dark" || stored === "system"
    ? stored
    : "light";
}

function getSystemTheme(): ResolvedTheme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function PreferencesProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(readLanguage);
  const [theme, setThemeState] = useState<Theme>(readTheme);
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(getSystemTheme);

  const resolvedTheme = theme === "system" ? systemTheme : theme;

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setSystemTheme(getSystemTheme());
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    localStorage.setItem(LANGUAGE_KEY, language);
  }, [language]);

  useEffect(() => {
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme;
    document.documentElement.lang = language;
  }, [language, resolvedTheme]);

  const t = useCallback(
    (key: TranslationKey) => messages[language][key] || messages.en[key],
    [language],
  );

  const value = useMemo(
    () => ({
      language,
      theme,
      resolvedTheme,
      setLanguage: setLanguageState,
      setTheme: setThemeState,
      t,
    }),
    [language, resolvedTheme, t, theme],
  );

  return (
    <PreferencesContext.Provider value={value}>
      {children}
    </PreferencesContext.Provider>
  );
}

export function usePreferences(): PreferencesContextValue {
  const ctx = useContext(PreferencesContext);
  if (!ctx) {
    throw new Error("usePreferences must be used within PreferencesProvider");
  }
  return ctx;
}
