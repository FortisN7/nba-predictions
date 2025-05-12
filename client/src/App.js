import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [preds, setPreds] = useState([]);
  const [games, setGames] = useState([]);

  useEffect(() => {
    async function load() {
      const [pRes, gRes] = await Promise.all([
        axios.get('http://localhost:4000/predictions/last7'),
        axios.get('http://localhost:4000/games/last7'),
      ]);
      setPreds(pRes.data);
      setGames(gRes.data);
    }
    load();
  }, []);

  // group preds by date (yyyy-mm-dd)
  const byDate = preds.reduce((acc, p) => {
    const date = p.date.split('T')[0];
    if (!acc[date]) acc[date] = [];
    acc[date].push(p);
    return acc;
  }, {});

  // newest-first
  const sortedDates = Object.keys(byDate)
    .sort((a, b) => new Date(b) - new Date(a));

  // today in same format
  const todayStr = new Date().toISOString().split('T')[0];

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 20 }}>
      <h1 style={{
        textAlign: 'center',
        marginBottom: 24,
        fontSize: 32
      }}>
        NBA Game Predictions
      </h1>

      {sortedDates.map(date => (
        <div key={date} style={{ marginBottom: 32 }}>
          <h2 style={{
            borderBottom: '2px solid #444',
            paddingBottom: 4,
            marginBottom: 16,
            color: '#555'
          }}>
            {date}
            {date === todayStr ? ' (Today)' : ''}
          </h2>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
            gap: 16
          }}>
            {byDate[date].map((p, i) => {
              const actual = games.find(
                g => g.date.split('T')[0] === date
                  && g.team === p.team
                  && g.team_opp === p.team_opp
              );

              const predictedWinner = p.prediction === 1 ? p.team : p.team_opp;

              let actualDisplay = 'No data yet';
              if (actual) {
                const homePts = actual.pts;
                const awayPts = actual.pts_opp;
                if (homePts > awayPts) {
                  actualDisplay = `${actual.team} (${homePts}-${awayPts})`;
                } else if (awayPts > homePts) {
                  actualDisplay = `${actual.team_opp} (${awayPts}-${homePts})`;
                } else {
                  actualDisplay = `Tie (${homePts}-${awayPts})`;
                }
              }

              return (
                <div
                  key={i}
                  style={{
                    border: '1px solid #ccc',
                    borderRadius: 8,
                    padding: 16,
                    background: '#fafafa',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                  }}
                >
                  <div style={{ fontWeight: 'bold', marginBottom: 8 }}>
                    {p.team} vs {p.team_opp}
                  </div>
                  <div style={{ marginBottom: 6 }}>
                    <strong>Prediction:</strong> {predictedWinner}
                  </div>
                  <div>
                    <strong style={{ fontStyle: 'italic' }}>Actual:</strong> {actualDisplay}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

export default App;
