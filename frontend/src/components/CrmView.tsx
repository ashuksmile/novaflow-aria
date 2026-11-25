import type { DashboardData } from '../types';

interface CrmViewProps {
  data: DashboardData | null;
  error: string | null;
  onRefresh: () => void;
}

export function CrmView({ data, error, onRefresh }: CrmViewProps) {
  const m = data?.metrics;
  const leads = data?.leads ?? [];

  return (
    <section id="view-crm" className="light">
      <div className="wrap">
        <div className="mono">CRM / PIPELINE / LIVE</div>
        <h2>
          Every call
          <br />
          leaves a <span className="boxed">record</span>
        </h2>
        <p className="lede">
          Leads Aria qualified and scored in real time — synced to the CRM as the call happens.
        </p>

        <div className="kpis">
          <div className="kpi">
            <div className="mono">Total calls</div>
            <div className="n" id="kpi-calls">
              {m?.total_calls ?? 0}
            </div>
          </div>
          <div className="kpi">
            <div className="mono">Qualified leads</div>
            <div className="n" id="kpi-qualified">
              {m?.qualified_leads ?? 0}
            </div>
          </div>
          <div className="kpi">
            <div className="mono">Booked demos</div>
            <div className="n" id="kpi-demos">
              {m?.booked_demos ?? 0}
            </div>
          </div>
          <div className="kpi">
            <div className="mono">Avg lead score</div>
            <div className="n" id="kpi-score">
              {m?.avg_lead_score ?? 0}
            </div>
          </div>
          <div className="kpi">
            <div className="mono">Avg sentiment</div>
            <div className="n" id="kpi-sentiment">
              {m ? m.avg_sentiment.toFixed(1) : '0.0'}
            </div>
          </div>
        </div>

        <div className="rec-section">
          <div>
            <div className="mono">PIPELINE</div>
            <h3>Leads</h3>
            <button className="refresh mono" onClick={onRefresh}>
              Refresh ↻
            </button>
          </div>
          <div id="crm-leads">
            {error ? (
              <div className="empty mono">Could not load records: {error}</div>
            ) : leads.length > 0 ? (
              leads.map((l) => (
                <div key={l.session_id || l.id} className="row">
                  <div>
                    <div className="t">
                      {l.name || l.session_id}
                      {l.company ? ` — ${l.company}` : ''}
                    </div>
                    <div className="s mono">
                      {l.qualification_stage} · SCORE {l.lead_score}/100 · {l.sentiment_label}{' '}
                      {Number(l.sentiment_score || 0).toFixed(2)} · {l.language || 'en'}
                      {l.email ? ` · ${l.email}` : ''}
                    </div>
                    <div className="d">
                      Need: {l.need || '—'} &nbsp;·&nbsp; Budget: {l.budget || '—'} &nbsp;·&nbsp;
                      Timeline: {l.timeline || '—'}
                    </div>
                  </div>
                  <span className="pill">{l.lead_score}</span>
                </div>
              ))
            ) : (
              <div className="empty mono">No leads captured yet.</div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
