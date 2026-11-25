import type { DashboardData } from '../types';

interface InsightsViewProps {
  data: DashboardData | null;
  error: string | null;
  onRefresh: () => void;
}

export function InsightsView({ data, error, onRefresh }: InsightsViewProps) {
  const summaries = data?.summaries ?? [];

  return (
    <section id="view-insights" className="light">
      <div className="wrap">
        <div className="mono">POST-CALL / INTELLIGENCE</div>
        <h2>
          Call <span className="boxed">insights</span>
        </h2>
        <p className="lede">
          Executive summaries, buyer objections, next actions and follow-up emails generated after
          each call.
        </p>

        <div className="rec-section">
          <div>
            <div className="mono">SUMMARIES</div>
            <h3>Call reports</h3>
            <button className="refresh mono" onClick={onRefresh}>
              Refresh ↻
            </button>
          </div>
          <div id="crm-summaries">
            {error ? (
              <div className="empty mono">Could not load insights: {error}</div>
            ) : summaries.length > 0 ? (
              summaries.map((s) => (
                <div key={s.session_id || s.id} className="row">
                  <div>
                    <div className="s mono">
                      {s.session_id} · SCORE {s.lead_score}/100 · {s.sentiment_trend}
                    </div>
                    <div className="t" style={{ fontSize: 20, marginTop: 6 }}>
                      {s.summary}
                    </div>
                    <div className="d">Next: {s.recommended_action}</div>
                    {s.followup_email && <div className="summary-mail">{s.followup_email}</div>}
                  </div>
                  <span className="arrow">↗</span>
                </div>
              ))
            ) : (
              <div className="empty mono">End a call to generate its summary here.</div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
