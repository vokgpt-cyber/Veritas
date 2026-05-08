/**
 * EPAM law firm logo component with VERITAS branding.
 *
 * Features:
 * - SVG "V" monogram icon in EPAM red with scale/balance motif
 * - Full name subtitle: "Verbal Intelligence, Transcription And Summarization"
 * - Three sizes: sm (sidebar), md (general), lg (login page)
 */

interface LogoProps {
  size?: "sm" | "md" | "lg";
  showSubtitle?: boolean;
}

/**
 * VERITAS "V" monogram icon — a stylized V with a balance/scale accent,
 * evoking legal precision. Pure SVG, no external assets.
 */
function VeritasIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      {/* Shield/badge background */}
      <path
        d="M16 2L4 8v8c0 7.73 5.12 14.96 12 16 6.88-1.04 12-8.27 12-16V8L16 2z"
        fill="currentColor"
        opacity="0.08"
      />
      {/* V letterform — bold, geometric */}
      <path
        d="M8 8l8 18 8-18"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      {/* Horizontal bar — balance/scale accent */}
      <path
        d="M6 8h20"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      {/* Small diamond accent at the base of V */}
      <path
        d="M16 24l-1.5-2h3L16 24z"
        fill="currentColor"
        opacity="0.5"
      />
    </svg>
  );
}

export default function Logo({ size = "md", showSubtitle = false }: LogoProps) {
  const config = {
    sm: {
      text: "text-lg",
      sub: "text-[10px]",
      byEpam: "text-xs",
      icon: "w-5 h-5",
      subtitle: "text-[8px]",
      gap: "gap-1.5",
    },
    md: {
      text: "text-2xl",
      sub: "text-sm",
      byEpam: "text-sm",
      icon: "w-7 h-7",
      subtitle: "text-[10px]",
      gap: "gap-2",
    },
    lg: {
      text: "text-4xl",
      sub: "text-base",
      byEpam: "text-base",
      icon: "w-10 h-10",
      subtitle: "text-xs",
      gap: "gap-2.5",
    },
  };

  const s = config[size];

  return (
    <div className="flex flex-col items-center select-none">
      <div className={`flex items-center ${s.gap}`}>
        <VeritasIcon className={`${s.icon} text-epam-red`} />
        <div className="flex items-baseline gap-1.5">
          <span
            className={`font-heading font-bold text-epam-red ${s.text} tracking-wide`}
          >
            VERITAS
          </span>
          <span className={`font-heading text-epam-gray-500 ${s.byEpam}`}>
            by EPAM
          </span>
        </div>
      </div>
      {showSubtitle && (
        <p
          className={`text-epam-gray-400 ${s.subtitle} mt-1 tracking-[0.15em] uppercase font-body`}
        >
          Verbal Intelligence, Transcription &amp; Summarization
        </p>
      )}
    </div>
  );
}
