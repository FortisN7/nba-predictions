import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [games, setGames] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:4000/data/predictions.csv')
      .then(res => {
        const rows = res.data.trim().split('\n').slice(1);
        const data = rows.map(r => {
          const [date,team,opp,pred] = r.split(',');
          return { date, team, opp, pred: +pred };
        });
        // last 7 unique dates
        const uniq = [...new Set(data.map(g=>g.date))].slice(-7);
        setGames(data.filter(g=>uniq.includes(g.date)));
      });
  }, []);

  return (
    <div style={{
      display:'grid',
      gridTemplateColumns:'repeat(7,1fr)',
      gap:16,
      padding:20
    }}>
      {games.map((g,i)=>(
        <div key={i} style={{
          border:'1px solid #ddd', padding:12, borderRadius:4
        }}>
          <div><strong>{g.date}</strong></div>
          <div>{g.team} vs {g.opp}</div>
          <div>Predicted: {g.pred===1?'Win':'Lose'}</div>
        </div>
      ))}
    </div>
  );
}

export default App;
