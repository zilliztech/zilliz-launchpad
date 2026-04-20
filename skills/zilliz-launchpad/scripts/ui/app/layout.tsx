import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "zilliz-launchpad",
  description: "Demo UI for your launchpad-generated search collection",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
