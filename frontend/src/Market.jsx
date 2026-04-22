import React, { useState } from "react";

// Popular stocks for dropdown suggestions
const STOCK_SUGGESTIONS = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corporation' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.' },
    { symbol: 'META', name: 'Meta Platforms Inc.' },
    { symbol: 'TSLA', name: 'Tesla Inc.' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation' },
    { symbol: 'JPM', name: 'JPMorgan Chase & Co.' },
    { symbol: 'V', name: 'Visa Inc.' },
    { symbol: 'UNH', name: 'UnitedHealth Group' },
    { symbol: 'JNJ', name: 'Johnson & Johnson' },
    { symbol: 'WMT', name: 'Walmart Inc.' },
    { symbol: 'PG', name: 'Procter & Gamble Co.' },
    { symbol: 'MA', name: 'Mastercard Inc.' },
    { symbol: 'BAC', name: 'Bank of America Corporation' },
    { symbol: 'VZ', name: 'Verizon Communications Inc.' },
    { symbol: 'UNH', name: 'UnitedHealth Group' },
    { symbol: 'HD', name: 'The Home Depot Inc.' },
    { symbol: 'DIS', name: 'The Walt Disney Company' },
    { symbol: 'CMCSA', name: 'Comcast Corporation' },
    { symbol: 'NFLX', name: 'Netflix Inc.' },
    { symbol: 'CRM', name: 'Salesforce Inc.' },
    { symbol: 'INTC', name: 'Intel Corporation' },
];

const Market = () => {
    const [query, setQuery] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [showSuggestions, setShowSuggestions] = useState(false);

    const handleInputChange = (e) => {
        const value = e.target.value.toUpperCase();
        setQuery(value);

        if (value.length > 0) {
            const filtered = STOCK_SUGGESTIONS.filter(stock =>
                stock.symbol.startsWith(value) || stock.name.toUpperCase().includes(value)
            );
            setSuggestions(filtered.slice(0, 5)); // Show max 5 suggestions
            setShowSuggestions(true);
        } else {
            setSuggestions([]);
            setShowSuggestions(false);
        }
    };

    const handleSelectStock = (symbol) => {
        setQuery(symbol);
        setShowSuggestions(false);
    };

    const handlePredict = () => {
        if (query.trim()) {
            console.log(`Predicting for: ${query}`);
            // TODO: Call API with the selected ticker
        }
    };

    return (
        <div style={{ position: 'relative' }}>
            <input
                className="input-box"
                placeholder="Enter ticker e.g. TSLA"
                value={query}
                onChange={handleInputChange}
                onFocus={() => query && setShowSuggestions(true)}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
            />
            {showSuggestions && suggestions.length > 0 && (
                <ul className="suggestions-dropdown">
                    {suggestions.map((stock) => (
                        <li
                            key={stock.symbol}
                            onClick={() => handleSelectStock(stock.symbol)}
                        >
                            <strong>{stock.symbol}</strong> - {stock.name}
                        </li>
                    ))}
                </ul>
            )}
            <button className="search-button" onClick={handlePredict}>Predict</button>
        </div>
    );
};

export default Market;