const React = require('react');
const { useState, useRef, useEffect } = React;
const { FaPaperPlane, FaArrowDown } = require('react-icons/fa');

const FinancialChatbot = () => {
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isScrolled, setIsScrolled] = useState(false);
  const messagesEndRef = useRef(null);

  const stockFullNames = {
    AAPL: 'Apple Inc.',
    MSFT: 'Microsoft Corporation',
    GOOG: 'Alphabet Inc.',
    AMZN: 'Amazon.com Inc.',
    TSLA: 'Tesla Inc.',
    NVDA: 'NVIDIA Corporation',
    // Add more mappings as needed
  };

  const handleInputChange = (event) => {
    setInputMessage(event.target.value);
  };

  const handleSendMessage = async () => {
    if (inputMessage.trim() === '') return;

    const userMessage = { sender: 'user', text: inputMessage };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    const stockQueryRegex = /stocks data from (\w+)/i;
    const match = inputMessage.match(stockQueryRegex);

    if (match) {
      const symbol = match[1].toUpperCase();
      const fullName = stockFullNames[symbol] || symbol;

      try {
        const startDate = '2024-10-01';
        const endDate = '2024-10-31';

        const response = await fetch('http://127.0.0.1:5000/get-stock-data', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            symbol,
            start_date: startDate,
            end_date: endDate,
          }),
        });

        const data = await response.json();

        if (response.ok) {
          const latestData = data.data[data.data.length - 1];
          const stockDetails = {
            Ticker: symbol,
            Price: `$${latestData.c || 'N/A'}`,
            Volume: latestData.v || 'N/A',
            Open: `$${latestData.o || 'N/A'}`,
            Close: `$${latestData.c || 'N/A'}`,
            High: `$${latestData.h || 'N/A'}`,
            Low: `$${latestData.l || 'N/A'}`,
            Change: `$$${(latestData.c - latestData.o).toFixed(2) || 'N/A'}`,
          };

          const botResponse = {
            sender: 'bot',
            text: `Stock prices for ${fullName} (${symbol}) from ${startDate} to ${endDate}:`,
            table: [stockDetails],
          };

          setMessages((prevMessages) => [...prevMessages, botResponse]);
        } else {
          const errorResponse = {
            sender: 'bot',
            text: `Error fetching stock data: ${data.error}`,
          };
          setMessages((prevMessages) => [...prevMessages, errorResponse]);
        }
      } catch (error) {
        const errorMessage = {
          sender: 'bot',
          text: `An error occurred: ${error.message}`,
        };
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
      }
    } else {
      const noResponse = {
        sender: 'bot',
        text: `I can only help with stock-related queries. Please ask about stocks like: "Stocks data from AAPL".`,
      };
      setMessages((prevMessages) => [...prevMessages, noResponse]);
    }

    setInputMessage('');
    scrollToBottom();
  };

  const handleClearMessages = () => {
    setMessages([]);
  };

  const handleScroll = () => {
    const isAtBottom =
      messagesEndRef.current &&
      messagesEndRef.current.getBoundingClientRect().bottom <= window.innerHeight;
    setIsScrolled(!isAtBottom);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    setIsScrolled(false);
  };

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return React.createElement(
    'div',
    {
      style: {
        backgroundColor: '#e6f7ff',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        fontFamily: 'Arial, sans-serif',
      },
    },
    React.createElement(
      'div',
      {
        style: {
          flex: 1,
          overflowY: 'auto',
          padding: '20px',
          marginBottom: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px',
        },
      },
      messages.map((message, index) =>
        React.createElement(
          'div',
          {
            key: index,
            style: {
              alignSelf: message.sender === 'user' ? 'flex-end' : 'flex-start',
              backgroundColor: message.sender === 'user' ? '#0066cc' : '#f1f1f1',
              color: message.sender === 'user' ? 'white' : 'black',
              padding: '10px',
              borderRadius: '10px',
              maxWidth: '70%',
              wordWrap: 'break-word',
              whiteSpace: 'pre-wrap',
            },
          },
          message.table
            ? React.createElement(
                'table',
                {
                  style: {
                    borderCollapse: 'collapse',
                    width: '100%',
                  },
                },
                React.createElement(
                  'thead',
                  null,
                  React.createElement(
                    'tr',
                    null,
                    Object.keys(message.table[0]).map((header, i) =>
                      React.createElement(
                        'th',
                        {
                          key: i,
                          style: {
                            border: '1px solid #ddd',
                            padding: '8px',
                            backgroundColor: '#f2f2f2',
                          },
                        },
                        header
                      )
                    )
                  )
                ),
                React.createElement(
                  'tbody',
                  null,
                  message.table.map((row, rowIndex) =>
                    React.createElement(
                      'tr',
                      { key: rowIndex },
                      Object.values(row).map((cell, cellIndex) =>
                        React.createElement(
                          'td',
                          {
                            key: cellIndex,
                            style: {
                              border: '1px solid #ddd',
                              padding: '8px',
                              textAlign: 'center',
                            },
                          },
                          cell
                        )
                      )
                    )
                  )
                )
              )
            : message.text
        )
      ),
      React.createElement('div', { ref: messagesEndRef })
    ),
    React.createElement(
      'div',
      {
        style: {
          display: 'flex',
          alignItems: 'center',
          padding: '10px',
          borderTop: '1px solid #ccc',
        },
      },
      React.createElement('input', {
        type: 'text',
        value: inputMessage,
        onChange: handleInputChange,
        placeholder: 'Type your message...',
        style: {
          flex: 1,
          padding: '8px',
          fontSize: '16px',
          borderRadius: '5px',
          border: '1px solid #0066cc',
          marginRight: '10px',
        },
      }),
      React.createElement(
        'button',
        {
          onClick: handleSendMessage,
          style: {
            backgroundColor: '#0066cc',
            color: 'white',
            border: 'none',
            borderRadius: '50%',
            width: '40px',
            height: '40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            marginRight: '10px',
          },
        },
        React.createElement(FaPaperPlane, { style: { fontSize: '18px' } })
      ),
      React.createElement(
        'button',
        {
          onClick: handleClearMessages,
          style: {
            backgroundColor: '#ff4d4d',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            padding: '10px',
            cursor: 'pointer',
          },
        },
        'Clear Messages'
      )
    ),
    isScrolled &&
      React.createElement(
        'button',
        {
          onClick: scrollToBottom,
          style: {
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            backgroundColor: '#0066cc',
            color: 'white',
            border: 'none',
            borderRadius: '50%',
            width: '50px',
            height: '50px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.1)',
          },
        },
        React.createElement(FaArrowDown, { style: { fontSize: '24px' } })
      )
  );
};

module.exports = FinancialChatbot;
