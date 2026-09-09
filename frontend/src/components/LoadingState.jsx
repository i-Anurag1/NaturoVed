import React from "react";

export default function LoadingState({ message = "Analyzing image and field context..." }) {
  return (
    <div className="loading-state">
      <div className="spinner" />
      <p>{message}</p>
    </div>
  );
}
