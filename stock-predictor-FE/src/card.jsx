import React from 'react';

export const Card = ({ className, ...props }) => {
  return (
    <div
      className={`bg-white shadow-lg rounded-lg ${className}`}
      {...props}
    />
  );
};

