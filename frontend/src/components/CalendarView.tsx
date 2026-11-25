import type { CalendarData } from '../types';

interface CalendarViewProps {
  data: CalendarData | null;
  error: string | null;
  onRefresh: () => void;
}

function formatDay(isoDate: string): { weekday: string; label: string } {
  const dt = new Date(`${isoDate}T00:00:00`);
  if (Number.isNaN(dt.getTime())) return { weekday: '', label: isoDate };
  return {
    weekday: dt.toLocaleDateString('en-US', { weekday: 'short' }),
    label: dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
  };
}

export function CalendarView({ data, error, onRefresh }: CalendarViewProps) {
  const days = data?.days ?? [];
  const appointments = data?.appointments ?? [];

  return (
    <section id="view-calendar" className="light">
      <div className="wrap">
        <div className="mono">CALENDAR / LIVE AVAILABILITY</div>
        <h2>
          Demo <span className="boxed">schedule</span>
        </h2>
        <p className="lede">
          Next five days of demo slots Aria checks and books live during calls.
        </p>

        {error ? (
          <div className="empty mono">Could not load calendar: {error}</div>
        ) : (
          <div className="cal-grid" id="cal-grid">
            {days.map((day) => {
              const { weekday, label } = formatDay(day.date);
              return (
                <div key={day.date} className="cal-day">
                  <div className="cal-day-head">
                    <div className="mono">
                      {weekday} · {day.date}
                    </div>
                    <div className="d-num">{label}</div>
                  </div>
                  <div className="cal-slots">
                    {day.slots.map((slot) => {
                      const appt = slot.status === 'Booked' ? slot.appointment : null;
                      return appt ? (
                        <div key={slot.time} className="slot booked">
                          <div className="mono">{slot.time} · BOOKED</div>
                          <div className="who">{appt.lead_name}</div>
                          <div className="co">{appt.company || appt.lead_email}</div>
                        </div>
                      ) : (
                        <div key={slot.time} className="slot">
                          <div className="mono">{slot.time}</div>
                          <div className="co">Open</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="rec-section">
          <div>
            <div className="mono">CONFIRMED</div>
            <h3>Booked demos</h3>
            <button className="refresh mono" onClick={onRefresh}>
              Refresh ↻
            </button>
          </div>
          <div id="crm-appointments">
            {appointments.length > 0 ? (
              appointments.map((a) => (
                <div key={a.id} className="row">
                  <div>
                    <div className="t">
                      {a.lead_name}
                      {a.company ? ` — ${a.company}` : ''}
                    </div>
                    <div className="s mono">
                      {a.slot_date} · {a.slot_time} · {a.lead_email} &nbsp;
                      <span className="pill y">{a.status}</span>
                    </div>
                    {a.notes && <div className="d">{a.notes}</div>}
                  </div>
                  <span className="arrow">↗</span>
                </div>
              ))
            ) : (
              <div className="empty mono">No demos booked yet — ask Aria to book one.</div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
