import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { err: null, info: null };
  }

  static getDerivedStateFromError(err) {
    return { err };
  }

  componentDidCatch(err, info) {
    this.setState({ info });
    // still log if possible
    console.error("[ErrorBoundary]", err, info);
  }

  render() {
    if (this.state.err) {
      return (
        <div style={{
          padding: 16,
          borderRadius: 12,
          background: "rgba(255,0,0,0.08)",
          border: "1px solid rgba(255,0,0,0.25)",
          color: "white",
          whiteSpace: "pre-wrap",
          fontSize: 12
        }}>
          <b>UI crashed:</b>{"\n"}
          {String(this.state.err)}{"\n\n"}
          {this.state.err?.stack || ""}
        </div>
      );
    }
    return this.props.children;
  }
}
