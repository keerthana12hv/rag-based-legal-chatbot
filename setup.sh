mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[theme]\n\
primaryColor = '#4CAF50'\n\
backgroundColor = '#0E1117'\n\
secondaryBackgroundColor = '#262730'\n\
textColor = '#FAFAFA'\n\
font = 'sans serif'\n\
" > ~/.streamlit/config.toml
