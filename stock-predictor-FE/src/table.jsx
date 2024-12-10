import React from 'react';

export const Table = ({ className, ...props }) => {
  return (
    <table
      className={`w-full border-collapse ${className}`}
      {...props}
    />
  );
};

